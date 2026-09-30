# oaiy-keystore

**K1**, the named-secret keystore of the OAIY desktop: work package **V-02** of `design/vault.final.md`, section 4.5.1. A library, not linked by the
desktop yet; the relay design (D1: `relay.token`, `relay.host_identity`, `endpoint.x25519.<plugin>`) and the apps design (V1: `link.credential`) can code
against the trait now. A workspace member, not a default member: `cargo test -p oaiy-keystore` runs it, and the `vault-linux` and `vault-windows` lanes of
`.github/workflows/ci.yml` run it (and `oaiy-crypto`) on both platforms and in the release gate.

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
protected, a locked file, a directory, a FIFO or a link in its place, a file larger than any secret, a keys folder that has vanished or been replaced, a mode that
is too loose, a value that another provider made. A caller that meets an error does not mint a new identity and does not treat the name as a first run.

**Names and FormLogic's ids.** The name rule is this crate's, not FormLogic's: a FormLogic identifier with an upper-case letter (or one longer than 80 characters)
is `KeyError::InvalidName`, it is not folded to lower case for you, and two ids that differ only in case must not become one file. A caller that keys a secret
by such an id chooses the name itself: lower-case hex of the id's bytes (`oaiy_crypto::kdf::hex_lower`) is a valid name for an id of up to 40 bytes, with a
prefix of its own (`endpoint.x25519.<hex>`), and the mapping is the caller's to keep stable.

## Providers

| Provider | Where | What protects the value | Choice |
|---|---|---|---|
| `windows-dpapi-file` | `<data>/keys/<name>.ks` = `"OAIYKS1" \|\| 0x01 \|\|` DPAPI blob | Windows DPAPI, user scope (never `LOCAL_MACHINE`: the blob's flags are asserted to be zero), no UI, the name bound in as entropy `SHA-256("oaiy-ks:1\|" + name)` | the default on Windows |
| `keyfile` | `<data>/keys/<name>.kf` = `"OAIYKF1" \|\| 0x01 \|\| tag \|\| u32be(len) \|\| value \|\| check` | file permissions only: `0600` in a `0700` directory, in directories that others cannot rename in; the loader refuses looser modes, links, FIFOs and another owner. **The weakest provider; the value is in the clear in that file. Refused on Windows** (no modes there, and no ACL is set): `ProviderChoice::KeyfileUnsafe`, `keyfile-unsafe-for-tests`, is the only way to it, and is for tests | by name on Unix (`OAIY_KEY_PROVIDER=keyfile`), for headless machines |
| `os-keyring` (Secret Service) | | | named, **not built**: it answers `ProviderUnavailable` and never falls back |

`Auto` (unset `OAIY_KEY_PROVIDER`) is DPAPI on Windows and an error everywhere else: no provider is chosen for you where none is strong, and a typo in the
variable is an error and never `Auto`. **No plaintext fallback.**

The name is bound into each blob: DPAPI entropy on Windows; on the keyfile, a tag inside the file (`SHA-256("oaiy-ks:1|" + name)`) that is checked after the
integrity check, so a copy is `WrongName` and damage is `Corrupt`. A blob moved to another name fails; it never yields another name's secret. Both formats are
pinned byte for byte by known answers that node computed (`codec::tests::the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`,
`keystore::a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read`), so that a later version reads the files of this one.

**`put`** validates the value, makes the provider's bytes, writes them to a temporary file (`.<name>.<16 hex>.tmp`, created new, `0600` on Unix, a name that can
never be a key, locked while it is written), flushes it, reads it back through the provider, compares, and only then, holding the folder's lock, renames it over the
old file and flushes the folder. A failure at any step leaves the previous value and no file of the failed attempt. A file is read into one buffer of exactly its
size, so no smaller copy of a secret is left in freed memory.

## One folder, several processes, other users: what the store does (review H-1, M-2, M-3, L-6, L-9)

The independent review of `5859fd8d` found the primitives sound and the keystore not: a reader could be told "never stored" for a key that exists, a folder swapped
in by another user was read as the real one, a FIFO hung a read for ever, and opening the store deleted another process's write in progress. The rules now are:

- **The folder is held, not looked up** (`src/keydir.rs`). On Unix the store opens the keys folder once (`O_NOFOLLOW`) and opens, creates, renames and removes every
  file **relative to that descriptor** (`openat`, `renameat`, `unlinkat` through `rustix`'s safe API; the crate stays `#![deny(unsafe_code)]` outside the DPAPI
  module). A file is opened `O_NOFOLLOW | O_NONBLOCK` and judged by `fstat` on the descriptor that will be read: a regular file, owned by this user, no group or
  other permission. Before every operation the folder is judged again (owner, mode) and compared by device and inode with the path: a folder that was replaced,
  moved or removed is an error. At open the directories **above** are judged by walking `..` from the descriptor: not another user's (root's is allowed), and not
  writable by group or others unless sticky (as `/tmp`). On systems whose umask leaves new folders `0775` the data folder's parent must be made `0755`: the store
  says which level refuses and never repairs a mode. On Windows the folder is held open without `FILE_SHARE_DELETE`, so neither it nor a folder above it can be
  renamed or removed while the store is open; a junction or symbolic link is refused for the folder and for every file (DPAPI included), and the folder is flushed
  after a rename and a removal. The reviewer's swap (1.1 million in 40 seconds) read 3,388 of the attacker's values and answered `None` 58,442 times; with a real
  second account (WSL, uid 65534) against this store: 224,332 swaps, 0 attacker values, 0 `None`, the rest errors (`unix::uid_swap_stress_...`, `#[ignore]`d, below).
- **An advisory lock orders readers and writers** (`<keys>/.lock`; `flock` on Unix, `LockFileEx` on Windows, through std's `File::lock`). A reader holds it shared
  and a change of what a name refers to (the rename of a put, the removal of a delete) holds it exclusively, so a reader never sees a name in the middle of a replace
  (on Windows a rename over an existing file does leave such a moment, and a reader in another process used to get `None`). Each operation opens the lock file for
  itself, so the lock orders threads of one process as well as processes, is released if the process dies, and is waited for at most ten seconds (then an error that
  names the lock: a caller never hangs).
- **A folder remembers its provider** (`<keys>/.provider`, the provider's identifier). Reopened with another provider it is `ProviderMismatch` before anything is read;
  a folder without the marker is adopted only if it holds no file of another provider's kind; and a file of the other provider's kind under a name makes `get`, `put`,
  `delete` and `list` of it an error (there are no two live values under one name, and a secret this provider cannot read is not reported absent). There is **no
  migrate call yet**: moving a folder from one provider to another has to read every secret with the old one and write it with the new one.
- **Opening removes debris only**: a file of exactly the shape `.<name>.<16 hex>.tmp`, older than a minute, that nobody has locked. A put in progress is young and
  locked; two processes that start together, or a thread that opens the store in a loop, never break a put.
- **`.lock` and `.provider` are bookkeeping**: they hold no secret and no name, `list` does not show them, and no test counts them as key files.

## The rules of 4.5.1, and the test that carries each

Tests marked (all) run on every provider the platform has: the keyfile on Unix and the unsafe keyfile name on Windows, and DPAPI on Windows. `store::tests::*` and
`keydir::tests::*` are unit tests (with hooks that act at named points of the code: where an attacker would swap a folder, or another process would open the store).

| Rule | Test |
|---|---|
| a round trip; overwrite; delete; list with a prefix; every size from 1 byte to 64 KiB (all) | `keystore::round_trip_overwrite_delete_and_list_for_every_provider` |
| `Ok(None)` is never stored (all) | `keystore::a_name_that_was_never_stored_is_none_not_an_error` |
| `Err` is could not read: junk, an empty file, every truncation, every single bit flipped, an extension, a directory in its place, a locked file, an oversized file (all) | `keystore::a_store_that_cannot_read_its_key_is_an_error_and_never_none`, `keystore::a_locked_file_is_an_error_and_a_locked_destination_makes_put_fail_and_keep_the_old_value` (Windows), `keystore::a_file_that_is_far_larger_than_any_secret_is_refused_without_being_read_into_memory`, `no_big_reads::a_file_far_larger_than_any_secret_is_refused_before_it_is_read_into_memory` (the error comes before the allocation: a counting allocator) |
| a file that cannot be opened is an error, never `None` (Unix: mode 0000; Windows: the locked file) | `keystore::unix::a_key_file_that_cannot_be_opened_is_an_error_and_never_none` |
| a file that changed while it was read is an error, not a truncated secret | `keydir::tests::a_file_that_grew_or_shrank_while_it_was_read_is_an_error_not_a_secret`, `keydir::tests::a_length_no_blob_can_have_is_refused_before_anything_is_read_or_allocated` |
| a keys folder that has gone is an error, not an empty store; one that is held cannot be removed (Windows) (all) | `keystore::a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| **a folder swapped in after the check is never read or written** (M-3) | `store::tests::a_folder_swapped_in_after_the_check_is_never_read_or_written_in_place_of_the_real_one` (Unix, by a hook), `keystore::unix::uid_swap_stress_the_victim_never_reads_the_attackers_value_and_never_gets_none` (`#[ignore]`, root) |
| a FIFO in place of a key file is refused at once and never blocks; a link, another owner (M-3) | `keystore::unix::a_fifo_in_place_of_a_key_file_is_refused_at_once_and_never_blocks`, `keystore::unix::a_symbolic_link_in_place_of_the_file_or_the_directory_is_refused`, `perm::tests::*` |
| the folders above the keys folder: not writable by others unless sticky, not another user's, judged as the file system has them (M-3) | `keystore::unix::a_folder_above_the_keys_folder_that_others_can_rename_in_is_refused_unless_it_is_sticky`, `keystore::unix::the_folders_above_are_those_of_the_real_folder_however_the_path_reaches_it`, `perm::tests::an_ancestor_that_others_can_rename_in_is_refused_unless_it_is_sticky` |
| the owner rule is applied at open, before each operation, to a file, to the lock and to the folders above | `store::tests::what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above`; with a real second account: `keystore::unix::a_file_a_folder_and_a_parent_that_a_real_other_user_owns_are_refused` (`#[ignore]`, root) |
| a junction or link where the folder or a file should be is refused (Windows, both providers) (L-9) | `keystore::windows::a_junction_in_place_of_the_keys_folder_is_refused_for_every_provider`, `keystore::windows::a_junction_or_a_file_in_place_of_a_key_file_is_an_error_not_none`, `keystore::windows::a_keys_path_that_is_a_file_is_refused` |
| a put and a delete flush the folder (L-9) | `store::tests::a_put_and_a_delete_flush_the_folder` |
| **a reader is never told that a key that exists was never stored** (H-1) | `store::tests::a_reader_waits_for_a_writer_in_the_middle_of_a_replace_and_is_never_told_that_nothing_is_stored`, `keystore::readers_never_see_a_name_that_other_processes_are_rewriting_as_missing_or_torn` (two writer processes, three reader threads), `store::tests::readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait`, `store::tests::a_put_does_not_hold_the_lock_while_it_writes` |
| a folder remembers its provider; the other provider's files are refused everywhere (M-2) | `keystore::a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another`, `keystore::a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value`, `keystore::the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there` |
| opening the store never deletes a put in progress (L-6) | `store::tests::opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else`, `store::tests::a_put_in_progress_survives_another_process_opening_the_store`, `store::tests::a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock`, `keystore::puts_do_not_fail_while_the_store_is_being_opened_again_and_again` |
| a blob moved to another name fails (all) | `keystore::a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret`, `dpapi::a_blob_that_names_a_master_key_this_user_does_not_hold_is_an_error` |
| the formats and the DPAPI scope are pinned | `codec::tests::the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`, `keystore::a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read`, `keystore::dpapi::the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` (flags zero) |
| another user's blob fails (Windows, `#[ignore]`) | `dpapi::another_users_blob_fails`: **not run**, it needs a blob made by another Windows account (below); its stand-in, a blob that names a master key this user does not hold, runs |
| no plaintext beside the blob: after a success, an overwrite, a failed put and a delete (all); the bookkeeping files hold none | `keystore::no_plaintext_beside_the_blob`, `dpapi::the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` |
| `put` reads back and compares, else `Err`; a failure leaves the previous value (all) | `store::tests::a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris`, `..::a_put_whose_read_back_fails_...`, `..::a_put_that_cannot_create_its_temporary_file_...` |
| keyfile mode refusal | `perm::tests::keyfile_mode_refusal_over_every_permission_bit_pattern` (all 4,096 patterns, every platform), and on Unix `unix::a_directory_or_file_with_any_group_or_other_permission_is_refused_and_never_repaired`, `unix::a_new_store_is_0700_and_its_files_are_0600` |
| names: the regular expression, the Windows device names (`con`, `nul`, `com1` ...), one plain path component | `name::tests::*` |
| fail closed: no provider chosen for you, a typo is an error, the Secret Service does not fall back | `keystore::the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| zeroization: every copy of a 30 KB secret is zero when it is freed, on `put`, `get` and a failed `get` (all) | `zeroize_probe::the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` (an allocator that reads each block as it is freed, with a control) |
| concurrent readers never see a torn value; no temporary survives (all) | `keystore::many_threads_reading_and_writing_never_see_a_torn_value` |

### The tests that need root or another account

A test cannot make a second user, and this crate does not create accounts. Three tests are `#[ignore]`d:

- `dpapi::another_users_blob_fails` (Windows), from a blob another account made: as a different Windows user `cargo run -p oaiy-keystore --example write_blob -- <folder> foreign.secret`;
  as the usual user `set OAIY_KS_FOREIGN_BLOB=<folder>\keys\foreign.secret.ks`, then `cargo test -p oaiy-keystore -- --ignored another_users_blob`. It has not been run.
- `unix::a_file_a_folder_and_a_parent_that_a_real_other_user_owns_are_refused` and `unix::uid_swap_stress_the_victim_never_reads_the_attackers_value_and_never_gets_none`
  (Unix, root): **build the test binary as yourself and run it as root**, because the victim is root and the attacker is uid 65534 (the stress re-runs the test binary as
  that user, which has to be able to read the path to it):

  ```text
  cargo test -p oaiy-keystore --test keystore --no-run            # prints   Executable tests/keystore.rs (target/debug/deps/keystore-<hash>)
  sudo target/debug/deps/keystore-<hash> --ignored --nocapture --test-threads=1
  # in WSL, from Windows:   wsl -d Ubuntu-24.04 -u root -- /tmp/oaiy-vault-wsl/target/debug/deps/keystore-<hash> --ignored --nocapture --test-threads=1
  ```

  Both ran on the final code in WSL2 Ubuntu 24.04 as root: the owner test passed, and the stress made 226,209 swaps in five seconds while the victim read 114,837
  values of its own, got 1,782,504 errors (the folder is not where it was: fail closed), no `None` and no attacker value. The stress also asserts that a new `open` in the
  world-writable parent is refused.

## Platforms, and what was actually run

| Platform | Result |
|---|---|
| Windows 11 (this machine): DPAPI and the unsafe keyfile | `cargo test --locked -p oaiy-crypto -p oaiy-keystore`: **oaiy-keystore 47 passed, 1 ignored** (the other-user DPAPI test); with oaiy-crypto 98 passed and 1 ignored: 145 and 2 |
| Linux (WSL2 Ubuntu 24.04, ext4, rustc 1.94, from a copy in the WSL file system, WSL stopped afterwards): keyfile, the `unix` tests, the two-process and FIFO tests, the allocator probe | **oaiy-keystore 53 passed, 2 ignored** (the two root tests, which also passed when run as root: above); with oaiy-crypto 151 and 3 |
| macOS | **compile only**: `cargo check --target aarch64-apple-darwin --all-targets` is clean (and is a step of the `vault-linux` lane); the keyfile provider is the only one there, and only by name |
| `x86_64-unknown-linux-musl` | `cargo clippy --all-targets -- -D warnings` clean (compiles the Unix code from Windows) |

## Unsafe

One module, `dpapi.rs`, Windows only: the two calls `CryptProtectData` and `CryptUnprotectData` (through `windows-sys`, bindings only) and `LocalFree`. The crate root is
`#![deny(unsafe_code)]`, the module carries the one `#[allow(unsafe_code)]`, and each block says why it is sound. `rustix` (Unix) is used through its safe API only. What the DPAPI module does
that a safe wrapper could not: it copies the plaintext DPAPI returns into a zeroizing buffer, overwrites the system's copy with volatile writes and only then frees it. That overwrite cannot be
observed from Rust (the memory is not the Rust allocator's); it is covered by review, and the mutant that removes it is reported as surviving.

The tests contain `unsafe` in two files, `tests/zeroize_probe.rs` and `tests/no_big_reads.rs`, and nowhere else: each is a `GlobalAlloc` that forwards every call to the system allocator (the first also reads a block
as it is freed, the second records the largest request), with the safety argument beside it.

## Dependencies

| Crate | Locked | Licence | Why |
|---|---|---|---|
| `oaiy-crypto` | path, 0.1.0 | Apache-2.0 | SHA-256 (the name binding and the integrity check), the constant-time `ct_eq`, the random generator; and its dependencies (46 crates, listed in its README) |
| `zeroize` | 1.9.0 | Apache-2.0 OR MIT | `Zeroizing<Vec<u8>>`, the return type of `get` |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | Windows only, features `Win32_Foundation` and `Win32_Security_Cryptography`: the DPAPI bindings. Already in the workspace lock (the tray app uses it), so it adds no package |
| `rustix` | 1.1.4 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | Unix only, features `std`, `fs`, `process`, through its safe API: the directory descriptor and the calls relative to it (`openat`, `renameat`, `unlinkat`, `fsync`, directory listing), which std does not offer, and `geteuid`. Already in the workspace lock, so it adds no package (`bitflags`, `errno`, `libc`, `linux-raw-sys`, its own dependencies, are in it too) |

No dev-dependencies. For the two crates together the workspace lock gained 39 `[[package]]` entries (`oaiy-crypto`, `oaiy-keystore` and 37 others, most of them the RustCrypto and dalek crates of `oaiy-crypto`'s tree, plus its dev-dependencies `bip39` and its own tree), and no existing package changed version: the edits to existing entries are that two dependency lines that said `rand_core` now say `rand_core 0.9.5`, because a second version is present, and one added line each for `rustix` (in `oaiy-keystore`) and `oaiy-crypto` (in its own dev-dependencies, for the `test-vectors` feature) (`git diff be3dd0bb..HEAD -- Cargo.lock`). `node platform/scripts/audit-vault-crates.mjs` audits the 75 packages of the two crates' tree with warnings denied (the CI lane runs it): clean.

## Where this differs from the design

1. **The Secret Service provider is not built** (the design allows "Secret Service **or** a keyfile"): it cannot be run or tested on this machine, adds a D-Bus stack to the lock, and a
   provider that has never run does not belong under a vault. `os-keyring` is a named choice that answers `ProviderUnavailable`.
2. **The keyfile format is mine** (`.kf`, tag, length, check): the design specifies only its modes. The DPAPI format is the design's.
3. **Names may not be Windows device names** (`con`, `prn`, `aux`, `nul`, `com0` to `com9`, `lpt0` to `lpt9`, judged before the first `.`), on every platform: `nul.ks` is the NUL
   device on Windows and the regular expression admits it (a design defect, below).
4. **Values are at least one byte** (DPAPI refuses an empty one, and an empty secret is not a secret).
5. **A keys folder that has gone is an error** on every operation (the design says `Err` never becomes "empty"; this applies it to the folder as well as the file).
6. **`Auto` never selects the keyfile**, and **the keyfile is refused on Windows** except under its unsafe-for-tests name (review M-2; the design allows a keyfile; on Windows
   nothing here could make it private, and setting an ACL needs more `unsafe`).
7. **The store holds its folder and locks it** (review H-1, M-3): a directory descriptor and calls relative to it on Unix, a handle that blocks renames on Windows, an
   advisory lock file, a provider marker. The design says only "`Ok(None)` is never stored" and "the loader refuses looser modes"; these are what it takes to make
   those true against another process and another user.
8. **Group-writable directories above the keys folder are refused** as well as world-writable ones: the store cannot know that a group has one member, so a data folder under a `0775`
   parent (the umask of some distributions) needs the parent made `0755`.

## Known limits

- Same-user malware defeats DPAPI (accepted, R1 of the design). An administrator's reset of the Windows password loses the DPAPI blobs: `get` answers an error, and everything K1 holds
  in Release 1 is re-derivable by reconnecting (4.5.2).
- Two processes writing the same name: each put is atomic and verified and readers never see `None` or a torn value; the last rename wins. A process that holds the lock and never lets go
  (or a file system with no locks: a network mount) makes an operation fail after ten seconds with an error that names the lock.
- **No migrate call**: a folder belongs to its provider. Switching providers is a copy that reads every secret with the old one and writes it with the new one, which does not exist yet.
- The directories above the keys folder are judged when the store is opened, not before each operation (a folder renamed later is caught by the comparison with the path, which is every operation).
  Windows trusts the held handle for the same guarantee. A path through `/mnt/c` under WSL (drvfs: every file `0777`) is refused by the ancestor rule; use the Linux file system.
- A crash between the flush and the rename leaves a temporary file until an `open` that finds it older than a minute and not locked.
- The plaintext that DPAPI allocates is overwritten by the module before it is freed, and that cannot be observed from a test (K23 below).

## Mutation checks

Each mutant breaks the code in one place (two where noted), runs the crate's whole suite, and restores the file from git; they are run in a second worktree so that the work goes on in the first:
`scratchpad/vault-impl/mutate2.ps1` and `mutants2.ps1` (kept outside the repository; the table is their output). **Round 1** (59 mutants of both crates at `5859fd8d`: 54 killed, five survivors explained,
the table of the previous edition of this file) was followed by the independent review, whose own mutants found ten places that no test noticed (M04, M33, M52, M53 and KM07, KM08, KM11, KM12, KM16, KM19, KM25). **Round 2**
is run on the code after the review's fixes, one mutant or more for each finding and each of those ten: the keystore's, below, and the crypto crate's in its README. `where` is Windows unless it says Linux (WSL).

**45 mutants: 44 killed, 1 survived** (B06, below). A and B are the folder held open and the lock; C the provider marker; D the debris; E and F and G the test gaps the reviewer's mutants found (E: reads; F: formats and scope; G: the owner rule).

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| A01 | a file is opened by path, not relative to the held folder (the swap attack works) | KILLED | `a_folder_swapped_in_after_the_check_is_never_read_or_written_in_place_of_the_real_one` |
| A02 | a key file is opened blocking (a FIFO hangs the read) | KILLED | `a_fifo_in_place_of_a_key_file_is_refused_at_once_and_never_blocks` |
| A03 | the descriptor of a key file is not judged (type, mode, owner) | KILLED | `a_directory_or_file_with_any_group_or_other_permission_is_refused_and_never_repaired`, `a_fifo_in_place_of_a_key_file_is_refused_at_once_and_never_blocks` |
| A04 | the directories above the keys folder are not judged | KILLED | `a_folder_above_the_keys_folder_that_others_can_rename_in_is_refused_unless_it_is_sticky`, `the_folders_above_are_those_of_the_real_folder_however_the_path_reaches_it` |
| A05 | only the first directory above is judged (the real chain is not walked) | KILLED | `the_folders_above_are_those_of_the_real_folder_however_the_path_reaches_it` |
| A06 | the path is not compared with the held folder before an operation | KILLED | `a_folder_swapped_in_after_the_check_is_never_read_or_written_in_place_of_the_real_one` |
| A07 | a put does not flush the folder after the rename | KILLED | `a_put_and_a_delete_flush_the_folder` |
| A08 | a delete does not flush the folder after the removal | KILLED | `a_put_and_a_delete_flush_the_folder` |
| A09 | a new temporary file is created readable by group and others | KILLED | `a_folder_swapped_in_after_the_check_is_never_read_or_written_in_place_of_the_real_one`, `a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris`, `a_put_and_a_delete_flush_the_folder` |
| A10 | the folder is held open with FILE_SHARE_DELETE (it can be renamed or removed under the store) | KILLED | `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| A11 | a junction or symbolic link is accepted for the folder and the files | KILLED | `a_junction_in_place_of_the_keys_folder_is_refused_for_every_provider` |
| A12 | a file where the keys folder should be is accepted | KILLED | `a_keys_path_that_is_a_file_is_refused` |
| B01 | a read takes no lock (it can see a name in the middle of a replace) | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait`, `a_reader_waits_for_a_writer_in_the_middle_of_a_replace_and_is_never_told_that_nothing_is_stored` |
| B02 | the rename of a put is not under the exclusive lock | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait` |
| B03 | the removal of a delete is not under the exclusive lock | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait` |
| B04 | list takes no lock | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait` |
| B05 | a read takes the lock exclusively (readers wait for each other) | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait` |
| B06 | the lock is not released by hand when it is dropped (only by closing the handle) | SURVIVED |  |
| C01 | the marker is read and not compared: a folder of another provider is opened | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another`, `the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there` |
| C02 | a folder with no marker is adopted although it holds another provider's files | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| C03 | get does not look for a value of another provider under the name | KILLED | `a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value` |
| C04 | put does not look for a value of another provider under the name (two live values) | KILLED | `a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value` |
| C05 | delete does not look for a value of another provider under the name | KILLED | `a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value` |
| C06 | list leaves out the other provider's secrets silently | KILLED | `a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value` |
| C07 | any text is a marker | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| C08 | the marker is never put in place (a new folder is not claimed) | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| C09 | the keyfile is accepted on Windows | KILLED | `the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there` |
| D01 | opening the store removes every temporary file, old or young, locked or not | KILLED | `a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock`, `a_put_in_progress_survives_another_process_opening_the_store`, `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else` |
| D02 | a young unlocked temporary file is debris (no age rule) | KILLED | `a_put_in_progress_survives_another_process_opening_the_store`, `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else`, `puts_do_not_fail_while_the_store_is_being_opened_again_and_again` |
| D03 | an old locked temporary file is debris (no lock rule) | KILLED | `a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock`, `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else` |
| D04 | a put does not lock its temporary file while it writes it | KILLED | `a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock` |
| D05 | any dot-file ending in .tmp counts as the store's temporary file | KILLED | `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else` |
| E01 | a file that grew during the read is not detected | KILLED | `a_file_that_grew_or_shrank_while_it_was_read_is_an_error_not_a_secret` |
| E02 | every open failure of a key file is read as never stored (Unix) | KILLED | `a_key_file_that_cannot_be_opened_is_an_error_and_never_none` |
| E03 | every open failure of a key file is read as never stored (Windows) | KILLED | `a_locked_file_is_an_error_and_a_locked_destination_makes_put_fail_and_keep_the_old_value`, `a_junction_or_a_file_in_place_of_a_key_file_is_an_error_not_none`, `a_store_that_cannot_read_its_key_is_an_error_and_never_none` |
| E04 | a file larger than any secret is read into memory | KILLED | `a_length_no_blob_can_have_is_refused_before_anything_is_read_or_allocated`, `a_file_far_larger_than_any_secret_is_refused_before_it_is_read_into_memory` |
| F01 | the name-binding domain string changes (every stored secret becomes unreadable) | KILLED | `the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`, `a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read` |
| F02 | the keyfile length is written little-endian | KILLED | `the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`, `a_put_does_not_hold_the_lock_while_it_writes`, `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait` |
| F03 | the keyfile length is read little-endian | KILLED | `the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`, `a_put_that_cannot_create_its_temporary_file_is_an_error_and_changes_nothing`, `a_put_does_not_hold_the_lock_while_it_writes` |
| F04 | DPAPI blobs are made for the machine, not the user | KILLED | `the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` |
| G01 | open does not compare the owner of the folder | KILLED | `what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above` |
| G02 | the owner of the folder is not compared before each operation | KILLED | `what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above` |
| G03 | the owner of a key file is not compared | KILLED | `what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above` |
| G04 | the owner of the lock file is not compared | KILLED | `what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above` |
| G05 | the owner of the folders above is not compared | KILLED | `what_another_user_owns_is_refused_at_open_before_each_operation_in_a_file_the_lock_and_the_folders_above` |

Where each of the reviewer's surviving mutants went (the round-1 mutants were **not** re-run on the refactored code: the code they broke is gone; the reviewer's survivors are covered by the round-2 mutants named here):

| Reviewer's survivor | Now |
|---|---|
| KM07: a `keys` path that is a file read as `None` under DPAPI | A12 (Windows), `unix::a_keys_path_that_is_a_file_is_refused`, `windows::a_keys_path_that_is_a_file_is_refused` |
| KM08: a file that grew during the read | E01 |
| KM11: the name-binding string | F01 |
| KM12: the keyfile length's byte order | F02 and F03 |
| KM16: the owner probe | G01 to G05 |
| KM19: the DPAPI scope | F04 |
| KM25: any failure to open read as never stored | E02 (Unix) and E03 (Windows) |

Two of the round's mutants survived at first and were killed only after their tests were extended, and that history is part of the result: **G01** (open does not compare the owner of the folder) because the folders above, judged with the same pretended user, refused first (the test now also opens a folder directly under `/tmp`); and, in the crypto crate, H05. **B01**, the lock, is killed only by the deterministic hook test (`a_reader_waits_for_a_writer_in_the_middle_of_a_replace...`); the two-process test that the coordinator asked for does not fail without the lock on this machine.

The survivors that cannot be observed from a test:

- **B06** does not unlock the folder's lock by hand when the guard is dropped: closing the handle releases it too, at once in every run, so nothing observes the difference (the unlock is kept: Windows documents that release at close
  is not always prompt).
- **K23** (round 1, still true) removes the overwrite of the plaintext that DPAPI allocated (`LocalAlloc`) before it is freed. That memory is not the Rust allocator's, so no probe in Rust can read it at the moment it is
  freed. It is covered by review of the `SAFETY` comment in `dpapi.rs`.
- **The flush itself** (round 1's K24, and `FlushFileBuffers` on the Windows folder handle): a test counts that the store asks for the flush (`a_put_and_a_delete_flush_the_folder`) but a power cut is not observable in-process.

A second finding of the round: the first version of the owner test (G01) passed even when `open` did not compare the owner, because the folders above, judged with the same pretended user, refused first. The test now also opens a folder directly under `/tmp`
(root's and sticky) so that nothing above it is what refuses, and G01 is killed. The two-process test did not reproduce H-1 without the lock on this machine in two runs of 28 seconds (the reviewer saw it with Defender running); the deterministic test is what carries the
finding, and the two-process test is what shows that two writer processes and three readers coexist.
