# oaiy-keystore

**K1**, the named-secret keystore of the OAIY desktop: work package **V-02** of `design/vault.final.md`, section 4.5.1. A library, not linked by the
desktop yet; the relay design (D1: `relay.token`, `relay.host_identity`, `endpoint.x25519.<plugin>`) and the apps design (V1: `link.credential`) can code
against the trait now. A workspace member, not a default member: `cargo test -p oaiy-keystore` runs it **as a product is built** (see the feature below), `cargo test -p oaiy-keystore
--features unsafe-keyfile` runs everything (on Windows, the keyfile tests too), and the `vault-linux` and `vault-windows` lanes of `.github/workflows/ci.yml` run both (and `oaiy-crypto`,
and the keystore's tests in a release build, where the unsafe name is refused too) on both platforms and in the release gate.

**The `unsafe-keyfile` feature** (off by default, and never to be enabled by a product or a dependency): adds `ProviderChoice::KeyfileUnsafe` (`keyfile-unsafe-for-tests`), the keyfile
provider on Windows, where nothing in this crate makes a file private and the plain `keyfile` is refused. Without it the name is `ProviderUnavailable` (from `ProviderChoice::parse` and from
`OAIY_KEY_PROVIDER`), the variant does not exist, and **on Windows the plaintext codec is not compiled in at all**, so no setting can make a Windows build store a value in the clear
(it used to be accepted from the environment in every build, and on a fresh Windows folder it made a plaintext store). The CI lanes check that the documentation of the library built
without the feature has no `KeyfileUnsafe`.

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
    fn put(&self, name: &Name, value: &[u8]) -> Result<Durability, KeyError>;    // atomic; reads back and compares. Err = nothing changed; Ok = stored
    fn delete(&self, name: &Name) -> Result<Durability, KeyError>;               // only after the caller verified its replacement. Err = still there; Ok = gone
    fn list(&self, prefix: &str) -> Result<Vec<Name>, KeyError>;
}
```

**`Err` from `put` or `delete` means that nothing changed, and `Ok` means that the change is made** (third review, M-A: over an SMB share every put and delete returned `Err` after the rename had
committed, so "a failed put leaves the previous value" was false there). What comes back with the `Ok` is a `Durability`: `Confirmed` (the folder was flushed after the rename, so the new name
survives a power cut) or `Unconfirmed` (the file system cannot flush a folder, or the flush failed: the value is stored and every reader sees it, and a power cut right after could bring the previous one
back; the file itself was flushed before it was renamed into place, so what comes back is whole). **A caller whose next step destroys the only other copy of what was replaced** (a rotation that has
re-wrapped data under the new key; a delete it cannot redo) treats `Unconfirmed` as "not safe yet": it keeps the old copy until a later change says `Confirmed` or the next start finds the new value there.
`store.put(..)?;` still compiles, and says nothing about durability: a caller that does not care does not look.

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
| `keyfile` | `<data>/keys/<name>.kf` = `"OAIYKF1" \|\| 0x01 \|\| tag \|\| u32be(len) \|\| value \|\| check` | file permissions only: `0600` in a `0700` directory, in directories that others cannot rename in; the loader refuses looser modes, links, FIFOs and another owner. **The weakest provider; the value is in the clear in that file. Refused on Windows** (no modes there, and no ACL is set): `ProviderChoice::KeyfileUnsafe`, `keyfile-unsafe-for-tests`, is the only way to it, exists only with the `unsafe-keyfile` feature, and is for tests | by name on Unix (`OAIY_KEY_PROVIDER=keyfile`), for headless machines |
| `os-keyring` (Secret Service) | | | named, **not built**: it answers `ProviderUnavailable` and never falls back |

`Auto` (unset `OAIY_KEY_PROVIDER`) is DPAPI on Windows and an error everywhere else: no provider is chosen for you where none is strong, and a typo in the
variable is an error and never `Auto`. **No plaintext fallback.**

The name is bound into each blob: DPAPI entropy on Windows; on the keyfile, a tag inside the file (`SHA-256("oaiy-ks:1|" + name)`) that is checked after the
integrity check, so a copy is `WrongName` and damage is `Corrupt`. A blob moved to another name fails; it never yields another name's secret. Both formats are
pinned byte for byte by known answers that node computed (`codec::tests::the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`,
`keystore::a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read`), so that a later version reads the files of this one.

**`put`** validates the value, makes the provider's bytes, writes them to a temporary file (`.<name>.<16 hex>.tmp`, created new, `0600` on Unix, a name that can
never be a key, locked while it is written), flushes it, reads it back through the provider, compares, and only then, holding the folder's lock, renames it over the
old file and flushes the folder. **A failure at any step up to and including the rename leaves the previous value and no file of the failed attempt** (a temporary file that cannot be removed is
debris, which the next open removes). The flush of the folder comes after the rename: it cannot undo it, so its failure is `Ok(Durability::Unconfirmed)` and not an error, on every platform. A file is
read into one buffer of exactly its size, so no smaller copy of a secret is left in freed memory.

## One folder, several processes, other users: what the store does (review H-1, M-2, M-3, L-6, L-9)

The independent review of `5859fd8d` found the primitives sound and the keystore not: a reader could be told "never stored" for a key that exists, a folder swapped
in by another user was read as the real one, a FIFO hung a read for ever, and opening the store deleted another process's write in progress. The rules now are:

- **The folder is held, not looked up** (`src/keydir.rs`). On Unix the store opens the keys folder once (`O_NOFOLLOW`) and opens, creates, renames and removes every
  file **relative to that descriptor** (`openat`, `renameat`, `unlinkat` through `rustix`'s safe API; the crate stays `#![deny(unsafe_code)]` outside the DPAPI
  module). A file is opened `O_NOFOLLOW | O_NONBLOCK` and judged by `fstat` on the descriptor that will be read: a regular file, owned by this user, no group or
  other permission. Before every operation the folder is judged again (owner, mode) and compared by device and inode with the path: a folder that was replaced,
  moved or removed is an error. At open the directories **above** are judged by walking `..` from the descriptor: not another user's (root's is allowed), and not
  writable by group or others unless sticky (as `/tmp`). On systems whose umask leaves new folders `0775` the data folder's parent must be made `0755`: the store
  says which level refuses and never repairs a mode. The reviewer's swap (1.1 million in 40 seconds) read 3,388 of the attacker's values and answered `None` 58,442
  times; with a real second account (WSL, uid 65534) against this store: 224,332 swaps, 0 attacker values, 0 `None`, the rest errors (`unix::uid_swap_stress_...`, `#[ignore]`d, below).
  **On Windows** (review M-1 corrected what an earlier edition of this file said): the folder is held open without `FILE_SHARE_DELETE`, so neither it nor a **real** folder above it
  can be renamed or removed while the store is open. That is **not** true of a junction or symbolic link above the folder: it is an entry in its parent that whoever can change the parent
  can delete and re-point with the store open, and the store then answered from the other tree (`None` for keys that exist, a write into the other tree, an older copy of the same
  user's keys served as the current value). So a path with a junction, a symbolic link or a mount point on it, **at any level**, is refused at open (walked before the folder is
  opened and again once it is held); every operation works in the held folder by its real path (the handle's final path: no junction in it, and it cannot change while the handle
  is open); before each operation the path is opened afresh and its volume and file index compared with the handle's; and every file is opened as itself and refused if it is a
  reparse point, not a regular file, or has more than one name (**on Unix too**: a key file has exactly one name, and a put repairs one that has two). The folder is flushed after a rename and a removal. A reparse point of any kind is refused, cloud placeholders
  (OneDrive) and a profile folder moved by junction included: fail closed; give the store the real path. **What is left, and is not claimed away:** whoever can write in the keys
  folder itself can delete a key file (a caller then sees `None`, "never stored") or put back an older copy of a DPAPI blob (which the same user can unprotect: a rollback). Windows
  gives `E:\` and other roots that nobody set up "Modify" to Authenticated Users, which is exactly that: **keep the data folder under the user's profile, on a local disk**
  (`%LOCALAPPDATA%`; not `%APPDATA%`, which folder redirection puts on a server: see "Network shares and redirected profiles"), whose access control is the user, SYSTEM and Administrators. The store does not read or set ACLs (a follow-up).
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
- **`.lock` and `.provider` are bookkeeping**: they hold no secret and no name, `list` does not show them, and no test counts them as key files. **They are looked for** (third review, L-5):
  the marker is read before every operation, and the lock file is made when the folder is opened and only opened afterwards, with one name like a key file. A store whose folder was emptied while
  it was open (`remove_dir_all` on a held folder on Windows deletes everything in it and then fails to remove the folder, which the store holds) used to answer `Ok(None)`, "never stored", for every key
  that had been there; now every operation is an error, whatever the name (`Corrupt`: the marker has gone; or `ProviderMismatch`: it names another provider; or the lock file could not be opened).
- **A flush that fails after the change is made is not an error of the change** (third review, M-A: see the next section). The first open ignores a failure to flush its marker.

## Network shares and redirected profiles (third review, M-A)

**What was found.** On an SMB share, `FlushFileBuffers` on a folder handle fails with os error 1 (`ERROR_INVALID_FUNCTION`, kind `Uncategorized`), which the Windows tolerance (the kinds `InvalidInput` and
`Unsupported`) never matched: the first open failed (with the lock file and the marker already written), and every `put` and `delete` returned `Err` **after** the rename or the removal had committed. A
caller that follows the contract after a failed rotation keeps the old key in memory while the new one is on disk and the old one is gone. It is reachable: the desktop's default data folder is under
`%APPDATA%` (`platform/desktop/src-tauri/src/lib.rs`), which domain folder redirection puts on a server.

**What is done.** The contract is stated exactly (above): `Err` means nothing changed, `Ok` means the change is made, and the flush that comes after the rename can only say `Confirmed` or `Unconfirmed`. Windows
knows os errors 1, 50 (`ERROR_NOT_SUPPORTED`) and 87 as "this file system cannot flush a folder", Unix `EINVAL`, `ENOTSUP` and `ENOSYS`; **every other failure to flush, after a change that is made, is
`Unconfirmed` too** (it cannot be an error), so an error code nobody has seen costs the report of durability and never correctness.

**What is supported, and what is claimed.** A data folder on an SMB share works: over loopback SMB (`\\localhost\c$`, the administrative share of this machine's `C:`, which is all that was tried), every step of a
first open, two puts (a rotation), a read from the other path of the same folder, a delete and an open again returns `Ok` with the right value, for DPAPI and for the plaintext provider alike, with
the advisory lock (`LockFileEx`) working between the share path and the local path of one folder (held exclusively through the local path, a read over the share waits ten seconds and gives up with the lock error, and works when it is let go: the test takes about twenty seconds), and every change `Unconfirmed`. **Not tested, and not claimed:** FAT, exFAT, ReFS, NFS, DFS, a share on another
machine, Offline Files, a roaming profile with a real server; each can answer a flush of a folder differently. **Recommended against**, all the same: a vault's keys belong on a local disk. A redirected
`%APPDATA%` is on a server whose administrators (and whoever can write to the share) can delete a key file (a caller sees `None`) or put back an older copy of a DPAPI blob (a rollback; the same user can unprotect
it), which is the "whoever can write in the keys folder" limit below, widened to everyone with access to the share. **A deployment that uses folder redirection should give the store
`%LOCALAPPDATA%` (never redirected) and not `%APPDATA%`**: the earlier editions of this file listed `%APPDATA%` beside `%LOCALAPPDATA%`, and the desktop's default data folder is there (`app_data_dir()`) (a follow-up for whoever links this store:
the desktop crate is not edited here).

**The opt-in test**, which needs the administrative share of the drive that the temp folder is on, and connects to nothing but `localhost`:

```text
cargo test -p oaiy-keystore --features unsafe-keyfile --test keystore -- --ignored smb --nocapture
```

It asks the share what a flush of a folder handle says, and requires every step to be `Ok` with the right value, the lock to hold between the two paths, and the durability to match (`Unconfirmed` where the flush fails, `Confirmed` where it works). Without
the fix its first put is an `Err` with the file on disk. The flush failures are also injected on every platform, for a put, a delete and a first open, by the unit tests of `store.rs`.

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
| a junction or link where the folder or a file should be is refused (Windows, both providers) (L-9), and a `keys` path that is a file is refused **as** "not a directory" (the kind is asserted: K17) | `keystore::windows::a_junction_in_place_of_the_keys_folder_is_refused_for_every_provider`, `keystore::windows::a_junction_or_a_file_in_place_of_a_key_file_is_an_error_not_none`, `keystore::windows::a_keys_path_that_is_a_file_is_refused_as_not_a_directory` |
| **a junction above the keys folder is refused, and nothing is made through it** (Windows, both providers) (M-1) | `keystore::windows::a_junction_above_the_keys_folder_is_refused_for_every_provider_and_nothing_is_made_through_it`, `keydir::sys::tests::a_junction_put_in_place_of_a_folder_between_the_walk_and_the_open_is_found_by_the_second_walk` (a hook puts the junction between the first walk and the open) |
| **the rollback through a re-pointed junction cannot happen**: the store that is open keeps answering with the current value, and a store opened through a junction to either copy is refused (M-1) | `keystore::windows::a_junction_pointed_at_an_older_copy_of_the_keys_folder_cannot_roll_a_value_back`, `keystore::windows::the_real_folders_above_the_keys_folder_cannot_be_renamed_or_replaced_while_the_store_is_open` |
| **a path that leads elsewhere is an error, never `None`, and every file is the held folder's** (M-1 (b) and (c)) | `keydir::sys::tests::a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders`, `store::tests::a_store_whose_path_leads_elsewhere_answers_with_errors_never_none_and_never_the_other_trees_value`; `winfs::tests::*` (the id, the final path, a path longer than the first buffer); the rules that judge a handle, over every attribute with made-up handles (a reparse point that is not a folder, a folder where a key file should be, a file with two names): `keydir::sys::tests::the_rules_for_a_folder_and_for_a_key_file_hold_over_every_attribute` |
| a key file has exactly one name (Windows and Unix) | `perm::tests::a_key_file_with_more_than_one_name_is_refused_and_a_folder_is_not_judged_by_its_count`, `keystore::unix::a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone`, `keystore::windows::a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| the unsafe keyfile is not in a build without the feature: refused from `parse` and from `OAIY_KEY_PROVIDER`, no plaintext provider on Windows, no folder made | `keystore::a_build_without_the_unsafe_keyfile_feature_refuses_the_unsafe_name_and_windows_has_no_plaintext_provider`, `keystore::the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` (run without `--features`); with the feature: `keystore::the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there` |
| a put and a delete flush the folder (L-9) | `store::tests::a_put_and_a_delete_flush_the_folder` |
| **a flush that fails after the change is made is not an error of the change: `Err` means nothing changed, `Ok` carries a `Durability`** (M-A), for a put, a delete and a first open, whether the file system cannot flush a folder or the flush failed | `store::tests::a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error`, `store::tests::the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed`, `store::tests::a_first_open_whose_flush_fails_still_opens_and_claims_the_folder`, `keydir::tests::only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so` (the os errors, per platform); over loopback SMB (opt-in, `#[ignore]`): `keystore::windows::a_store_on_a_share_over_loopback_smb_keeps_the_contract_whatever_the_share_says_about_flushing` |
| **a store whose folder was emptied while it was open answers with errors, never `None`; a lock file that has gone is an error and is not made again** (L-5) | `store::tests::a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none`, `store::tests::a_lock_file_that_has_gone_is_an_error_and_is_not_made_again`, `keystore::a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` (its Windows branch is the `remove_dir_all` case) |
| the lock file has one name and is not a link | `store::tests::a_lock_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` (both platforms), `store::tests::a_symbolic_link_in_place_of_the_lock_file_is_refused` (Unix) |
| a link to nothing under another provider's extension is still that provider's secret, not `None` | `store::tests::a_link_to_nothing_under_another_providers_extension_is_still_a_value_of_that_provider` (a symbolic link; a junction whose target is gone) |
| a key file that is a file symbolic link (a file reparse point) is refused as that, not followed or opened | `keystore::windows::a_file_symbolic_link_in_place_of_a_key_file_is_refused_not_followed`: **needs `SeCreateSymbolicLinkPrivilege`**, ends with a message where it is not held, and **has not been run on the machine that wrote it** (an account without the privilege); the hosted Windows runners of the CI lane hold it. And, opt-in because it starts WSL, with a reparse point that WSL makes without a privilege (`ln -s` on a Windows drive): `keystore::windows::a_file_reparse_point_made_by_wsl_in_place_of_a_key_file_is_refused_as_a_link_not_opened` (`cargo test -p oaiy-keystore --test keystore -- --ignored wsl`), which **was run**, and is what kills mutant Y16 |
| `list` is sorted whatever order the file system keeps; the debris shape is exactly 16 hex; a marker needs an identifier | `store::tests::list_is_sorted`, `store::tests::a_file_with_a_longer_random_part_than_the_stores_is_not_its_temporary_file`, `keystore::a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| **a reader is never told that a key that exists was never stored** (H-1) | `store::tests::a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none` (**the guard**: the real put paused inside its locked rename step, on every platform), `store::tests::a_reader_waits_for_a_writer_in_the_middle_of_a_replace_and_is_never_told_that_nothing_is_stored`, `keystore::readers_never_see_a_name_that_other_processes_are_rewriting_as_missing_or_torn` (two writer processes, three reader threads: a weak guard, see the limits), `store::tests::readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait`, `store::tests::a_put_does_not_hold_the_lock_while_it_writes` |
| a folder remembers its provider; the other provider's files are refused everywhere (M-2) | `keystore::a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another`, `keystore::a_secret_of_another_providers_kind_is_an_error_everywhere_never_none_and_never_a_second_value` |
| opening the store never deletes a put in progress (L-6) | `store::tests::opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else`, `store::tests::a_put_in_progress_survives_another_process_opening_the_store`, `store::tests::a_slow_write_that_is_older_than_a_minute_is_not_debris_while_its_writer_holds_the_lock`, `keystore::puts_do_not_fail_while_the_store_is_being_opened_again_and_again` |
| a blob moved to another name fails (all) | `keystore::a_blob_copied_or_moved_to_another_name_fails_and_never_yields_the_other_secret`, `dpapi::a_blob_that_names_a_master_key_this_user_does_not_hold_is_an_error` |
| the formats and the DPAPI scope are pinned | `codec::tests::the_name_binding_and_the_keyfile_format_are_pinned_byte_for_byte`, `keystore::a_keyfile_is_the_pinned_format_on_disk_and_a_pinned_file_is_read`, `keystore::dpapi::the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` (flags zero) |
| another user's blob fails (Windows, `#[ignore]`) | `dpapi::another_users_blob_fails`: **not run**, it needs a blob made by another Windows account (below); its stand-in, a blob that names a master key this user does not hold, runs |
| no plaintext beside the blob: after a success, an overwrite, a failed put and a delete (all); the bookkeeping files hold none | `keystore::no_plaintext_beside_the_blob`, `dpapi::the_file_is_a_dpapi_blob_with_the_magic_and_no_plaintext` |
| `put` reads back and compares, else `Err`; **a failure before the rename leaves the previous value** (a failure after it is `Ok(Unconfirmed)`: the row above) (all) | `store::tests::a_put_whose_read_back_differs_fails_leaves_the_old_value_and_no_debris`, `..::a_put_whose_read_back_fails_...`, `..::a_put_that_cannot_create_its_temporary_file_...` |
| keyfile mode refusal | `perm::tests::keyfile_mode_refusal_over_every_permission_bit_pattern` (all 4,096 patterns, every platform), and on Unix `unix::a_directory_or_file_with_any_group_or_other_permission_is_refused_and_never_repaired`, `unix::a_new_store_is_0700_and_its_files_are_0600` |
| names: the regular expression, the Windows device names (`con`, `nul`, `com1` ...), one plain path component | `name::tests::*` |
| fail closed: no provider chosen for you, a typo is an error, the Secret Service does not fall back | `keystore::the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| zeroization: every copy of a 30 KB secret is zero when it is freed, on `put`, `get` and a failed `get` (all) | `zeroize_probe::the_keystores_copies_of_a_secret_are_zero_when_they_are_freed` (an allocator that reads each block as it is freed, with a control) |
| concurrent readers never see a torn value; no temporary survives (all) | `keystore::many_threads_reading_and_writing_never_see_a_torn_value` |

### The tests that need root or another account

A test cannot make a second user, and this crate does not create accounts. Three tests are `#[ignore]`d for that (and two more are opt-in for other reasons, and `#[ignore]`d too: the one over loopback SMB, in "Network shares and redirected profiles", and the one that makes a file reparse point through WSL, in the table above):

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

  Both ran on the final code in WSL2 Ubuntu 24.04 as root (rustc 1.94, from a copy in the WSL file system, run from `target/debug/deps/keystore-<hash>`): the owner test passed, and the stress made 238,405 swaps in
  about five seconds while the victim read 119,760 values of its own (each read now also reads the marker), got 1,567,045 errors (the folder is not where it was: fail closed), no `None` and no attacker value. The stress also asserts that a new `open` in the
  world-writable parent is refused.

## Platforms, and what was actually run

| Platform | Result |
|---|---|
| Windows 11 (this machine): DPAPI and the unsafe keyfile | `cargo test --locked --no-fail-fast -p oaiy-crypto -p oaiy-keystore` (as a product is built): **oaiy-keystore 73 passed, 3 ignored** (the other-user DPAPI test, the opt-in SMB test and the opt-in WSL test; before the third review 60 and 1, before the second 47 and 1), oaiy-crypto 101 passed and 1 ignored (100 and 1; 98 and 1): 174 and 4. With `--features unsafe-keyfile` the keystore is also 73 and 3. `cargo test --release -p oaiy-keystore` (no feature): 73 and 3, the refusal of the unsafe name included. The opt-in tests were run: `--ignored smb` (loopback SMB over `\\localhost\c$`, both providers: every step `Ok`, every durability `Unconfirmed`) and `--ignored wsl` (a file reparse point made by WSL: refused as a link) |
| Linux (WSL2 Ubuntu 24.04, ext4, rustc 1.94, from a copy in the WSL file system, WSL stopped afterwards): keyfile, the `unix` tests, the two-process and FIFO tests, the allocator probe | **oaiy-keystore 67 passed, 2 ignored** (the two root tests, which also passed when run as root: above; before the third review 56 and 2, before the second 53 and 2), oaiy-crypto 101 and 1 (100 and 1; 98 and 1): 168 and 3. The same with `--features unsafe-keyfile` (67 and 2), and in a release build (`cargo test --release -p oaiy-keystore`: 67 and 2) |
| macOS | **compile only**: `cargo check --target aarch64-apple-darwin --all-targets` is clean (and is a step of the `vault-linux` lane); the keyfile provider is the only one there, and only by name |
| `x86_64-unknown-linux-musl` | `cargo clippy --all-targets -- -D warnings` clean (compiles the Unix code from Windows) |

## Unsafe

Two modules, both Windows only, both through `windows-sys` (bindings only). `dpapi.rs`: the two calls `CryptProtectData` and `CryptUnprotectData`, and `LocalFree`. `winfs.rs` (review M-1):
`GetFileInformationByHandle` and `GetFileInformationByHandleEx` (which file a handle is: volume and file index, how many names it has, its attributes) and `GetFinalPathNameByHandleW` (where it
really is), which `std` answers only on nightly (`windows_by_handle`) or not at all; each on a handle that a `&File` keeps open for the call, with a buffer the module owns. The crate root is
`#![deny(unsafe_code)]`, each of the two modules carries its one `#[allow(unsafe_code)]`, and each block says why it is sound. `rustix` (Unix) is used through its safe API only. What the DPAPI module does
that a safe wrapper could not: it copies the plaintext DPAPI returns into a zeroizing buffer, overwrites the system's copy with volatile writes and only then frees it. That overwrite cannot be
observed from Rust (the memory is not the Rust allocator's); it is covered by review, and the mutant that removes it is reported as surviving.

The tests contain `unsafe` in two files, `tests/zeroize_probe.rs` and `tests/no_big_reads.rs`, and nowhere else: each is a `GlobalAlloc` that forwards every call to the system allocator (the first also reads a block
as it is freed, the second records the largest request), with the safety argument beside it.

## Dependencies

| Crate | Locked | Licence | Why |
|---|---|---|---|
| `oaiy-crypto` | path, 0.1.0 | Apache-2.0 | SHA-256 (the name binding and the integrity check), the constant-time `ct_eq`, the random generator; and its dependencies (46 crates, listed in its README) |
| `zeroize` | 1.9.0 | Apache-2.0 OR MIT | `Zeroizing<Vec<u8>>`, the return type of `get` |
| `windows-sys` | 0.61.2 | MIT OR Apache-2.0 | Windows only, features `Win32_Foundation`, `Win32_Security_Cryptography` (the DPAPI bindings) and `Win32_Storage_FileSystem` (the file id, link count and final path of a handle). Already in the workspace lock (the tray app uses it), so it adds no package |
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
6. **`Auto` never selects the keyfile**, and **the keyfile is refused on Windows**, and is there only in a build with the `unsafe-keyfile` feature, under its unsafe-for-tests name (review M-2 and
   the second review; the design allows a keyfile; on Windows nothing here could make it private, and setting an ACL needs more `unsafe`).
7. **The store holds its folder and locks it** (review H-1, M-3): a directory descriptor and calls relative to it on Unix, a handle that blocks renames on Windows, an
   advisory lock file, a provider marker. The design says only "`Ok(None)` is never stored" and "the loader refuses looser modes"; these are what it takes to make
   those true against another process and another user.
8. **Group-writable directories above the keys folder are refused** as well as world-writable ones: the store cannot know that a group has one member, so a data folder under a `0775`
   parent (the umask of some distributions) needs the parent made `0755`.
9. **`put` and `delete` return a `Durability`** (third review, M-A), not `()`: the design says only "atomic" and "a failure leaves the previous value", which cannot be true of a failure that comes after the rename.
   Nothing links the trait yet, and `store.put(..)?;` still compiles. A caller that rotates a key should look at it (see the contract above).

## Known limits

- **`Durability::Unconfirmed` is a fact about the file system, and the store cannot repair it.** On a share, and on any file system that cannot flush a folder, a power cut just after a put could bring the previous value back (whole: the file is flushed
  before it is renamed), and just after a delete could bring the file back. Over loopback SMB every change is `Unconfirmed`. FAT, exFAT, ReFS, NFS, DFS and a share on another machine were **not tested** (see "Network shares and redirected
  profiles"); the classification knows os errors 1, 50 and 87 (Windows) and `EINVAL`, `ENOTSUP`, `ENOSYS` (Unix), and any other failure to flush after a change is `Unconfirmed` too.
- Same-user malware defeats DPAPI (accepted, R1 of the design). An administrator's reset of the Windows password loses the DPAPI blobs: `get` answers an error, and everything K1 holds
  in Release 1 is re-derivable by reconnecting (4.5.2).
- Two processes writing the same name: each put is atomic and verified and readers never see `None` or a torn value; the last rename wins. A process that holds the lock and never lets go
  (or a file system with no locks: a network mount) makes an operation fail after ten seconds with an error that names the lock.
- **No migrate call**: a folder belongs to its provider. Switching providers is a copy that reads every secret with the old one and writes it with the new one, which does not exist yet.
- The directories above the keys folder are judged when the store is opened, not before each operation (a folder renamed later is caught by the comparison with the path, which is every operation).
  On Windows the real folders above are held by the handle inside them and the path is compared with the handle before each operation; the window between the first walk for junctions
  and the open is closed by a second walk and the comparison, except for an attacker who flips the path back and forth faster than a check takes (every later operation compares again). A path through `/mnt/c` under WSL (drvfs: every file `0777`) is refused by the ancestor rule; use the Linux file system.
- A crash between the flush and the rename leaves a temporary file until an `open` that finds it older than a minute and not locked. That rule is tested at the minute on both platforms
  (`store::tests::opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else`, which runs on Windows: a file of 50 seconds survives and one of 70 is removed).
- **The lock wait is ten seconds** (`LOCK_WAIT`): one stall under Windows Defender in the reviewer's runs reached 8.5 seconds. An operation that gives up is an error that names the lock, fails
  closed and changes nothing, so a slow machine costs a caller a retry and never a wrong answer.
- **The two-process test is a weak guard of H-1**, and says so: the window is microseconds wide (the reviewer's lock-less store gave one false `None` in about 2.73 million reads and the locked
  store none in 2.0 million, which proves little). The guard is deterministic: `store::tests::a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none` pauses the real
  `put` inside its locked rename step and requires that a reader waits and never returns `None`.
- **Reparse points are refused at any level on Windows**, cloud placeholders (OneDrive) and a profile folder that was moved by a junction included: fail closed, give the store the real path.
  The store does not read or set ACLs; the data folder belongs under the user's profile, not under `E:\` or another root that nobody set up (see "What the store does").
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

### Round 3, after the second review

The second review found one medium (M-1, a junction above the keys folder on Windows) and lows. The mutants below break the code that fixed them, one place each, run the crate's suite (Windows, except R13 and R14, which are run on Linux under WSL; the ones that need the keyfile run with `--features unsafe-keyfile`, U01 and U03 without it) and restore the file from git (`scratchpad/vault-impl/mutate3.ps1` and `mutants3.ps1`, outside the repository; the crypto crate's half, the stack scrub and the in-place derivations, is in its README). **22 mutants of the keystore: 22 killed, none survived.**

| R01 | the path is not walked before the folder is made (a folder is made through a junction above it) | KILLED | `a_junction_above_the_keys_folder_is_refused_for_every_provider_and_nothing_is_made_through_it` |
| R02 | the path is not walked again once the folder is held (a junction put on the path between the first walk and the open is kept) | KILLED | `a_junction_put_in_place_of_a_folder_between_the_walk_and_the_open_is_found_by_the_second_walk` |
| R03 | a reparse point is not refused (junction, symbolic link, mount point) | KILLED | `the_rules_for_a_folder_and_for_a_key_file_hold_over_every_attribute` |
| R04 | the walk does not look at the folders on the way (only the keys folder is judged) | KILLED | `a_junction_put_in_place_of_a_folder_between_the_walk_and_the_open_is_found_by_the_second_walk`, `a_junction_above_the_keys_folder_is_refused_for_every_provider_and_nothing_is_made_through_it`, `a_junction_pointed_at_an_older_copy_of_the_keys_folder_cannot_roll_a_value_back` |
| R05 | the path is not compared with the held folder (volume and file index) | KILLED | `a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders` |
| R06 | verify does not compare the path with the handle before an operation | KILLED | `a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders` |
| R07 | files are reached by the path the caller gave, not the real path of the held folder | KILLED | `a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders` |
| R08 | a listing is of the path the caller gave, not of the held folder | KILLED | `a_path_that_leads_elsewhere_is_an_error_and_every_file_is_still_the_held_folders` |
| R09 | a file with more than one name is accepted (Windows) | KILLED | `a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| R10 | a file where the keys folder should be is accepted (K17) | KILLED | `a_keys_path_that_is_a_file_is_refused_as_not_a_directory` |
| R11 | the folder is held open with FILE_SHARE_DELETE | KILLED | `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| R12 | a file that is a directory is accepted as a key file | KILLED | `the_rules_for_a_folder_and_for_a_key_file_hold_over_every_attribute` |
| R13 | a key file with more than one name is accepted (Unix) | KILLED | `a_key_file_with_more_than_one_name_is_refused_and_a_folder_is_not_judged_by_its_count`, `a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| R14 | the link rule is off by one (two names are accepted) | KILLED | `a_key_file_with_more_than_one_name_is_refused_and_a_folder_is_not_judged_by_its_count`, `a_key_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| U01 | a build without the feature accepts the unsafe name as the plain keyfile | KILLED | `a_build_without_the_unsafe_keyfile_feature_refuses_the_unsafe_name_and_windows_has_no_plaintext_provider`, `the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| U02 | the plain keyfile is accepted on Windows | KILLED | `the_keyfile_is_refused_on_windows_and_the_unsafe_name_is_needed_to_use_it_there` |
| U03 | the environment variable names the unsafe keyfile and it is accepted without the feature | KILLED | `the_provider_is_chosen_on_purpose_and_a_typo_is_an_error` |
| P01 | the rename of a put is not under the exclusive lock (the hook is never reached) | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait`, `a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none`, `readers_never_see_a_name_that_other_processes_are_rewriting_as_missing_or_torn` |
| P02 | the lock is taken and released before the rename (a reader is not held) | KILLED | `a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none` |
| P03 | a read takes no lock | KILLED | `readers_share_the_lock_and_a_holder_that_never_lets_go_is_an_error_after_the_wait`, `a_reader_waits_for_a_writer_in_the_middle_of_a_replace_and_is_never_told_that_nothing_is_stored`, `a_reader_that_starts_in_the_middle_of_a_puts_rename_waits_and_never_returns_none` |
| P04 | a young temporary file is debris after five seconds, not after a minute | KILLED | `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else` |
| P05 | a temporary file is debris only after ten minutes | KILLED | `opening_the_store_removes_the_debris_of_an_interrupted_put_and_nothing_else` |

How this round went, because two of the mutants were alive at first: **R03** (a reparse point is not refused) and **R12** (a directory is accepted as a key file) survived the first run, because a junction is also a directory and a directory in place of a key file is refused by the open of the file before the rule is reached, so the junction tests and the store tests cannot tell whether the rules are there. They are killed by `the_rules_for_a_folder_and_for_a_key_file_hold_over_every_attribute`, which judges made-up handles over every combination of attribute bits (a file symbolic link and a cloud placeholder cannot be made in a test without privileges). A variant of R02 that removed a *third* walk of the path (between the open and the comparison) survived, because it was redundant: the walk was removed from the code (`956082a1`). U01 and U02 did not compile in their first form and were rewritten. The survivors that nothing can observe (B06, K23, the flush) are those of round 2, above, and are unchanged.

### Round 4, after the third review

The third review found one medium (M-A, a flush that fails after the change is made on a network share) and lows. The mutants below break the code that fixed M-A and L-5, and are the reviewer's survivors that this crate owns, Y06, Y07, Y10, Y13 to Y16, Y18, Y51 and Y54, run on Windows (A, B and most of the Y mutants) and on Linux under WSL (those marked as such: A07l, A08, B06u, Y18l, Y51, Y54), each with the whole keystore suite and `--features unsafe-keyfile` (`scratchpad/vault-impl/mutate4.ps1` and `mutants4.ps1`, outside the repository; Y16w with `-- --ignored wsl`). **32 new mutants of the keystore: 30 killed, 2 survived** (Y16 and Y18, below). Round 3's R05 to R09, U01 to U03 and P01 to P03 were run again, because the code they break had moved: all killed.

| # | Break | Result | Killed by (up to three tests) |
|---|---|---|---|
| A01 | Windows: os error 1 (an SMB share) is not a file system that cannot flush a folder | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so`, `the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed` |
| A02 | a put returns the error of the flush that comes after the rename (the old behaviour) | KILLED | `a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error` |
| A03 | a delete returns the error of the flush that comes after the removal | KILLED | `a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error` |
| A04 | the first open fails when the flush of its marker fails | KILLED | `a_first_open_whose_flush_fails_still_opens_and_claims_the_folder` |
| A05 | a flush that failed after the change is reported as Confirmed | KILLED | `a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error` |
| A06 | a file system that cannot flush a folder is reported as Confirmed | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so`, `the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed`, `a_flush_that_fails_after_the_change_is_made_does_not_make_a_put_or_a_delete_an_error` |
| A07 | every failure to flush the folder is Unconfirmed, none is an error (Y12) | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so`, `the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed` |
| A07l | every failure to flush the folder is Unconfirmed, none is an error (Y53, Linux) | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so`, `the_flush_of_a_folder_tells_a_file_system_that_cannot_from_a_flush_that_failed` |
| A08 | Unix: ENOTSUP is not a file system that cannot flush a folder | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so` |
| A09 | Windows: os error 50 (not supported) is not a file system that cannot flush a folder | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so` |
| A10 | Windows: access denied (5) counts as a file system that cannot flush a folder | KILLED | `only_the_errors_that_say_a_file_system_cannot_flush_a_folder_are_told_so` |
| B01 | a get does not look for the marker | KILLED | `a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none`, `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| B02 | a put does not look for the marker | KILLED | `a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none`, `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| B03 | a delete does not look for the marker | KILLED | `a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none`, `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| B04 | a list does not look for the marker | KILLED | `a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none`, `a_keys_directory_that_has_gone_is_an_error_not_an_empty_store` |
| B05 | the marker that an open store looks for may name another provider | KILLED | `a_store_whose_folder_was_emptied_while_it_was_open_answers_with_errors_never_none` |
| B06 | Windows: an operation makes the lock file again if it has gone | KILLED | `a_lock_file_that_has_gone_is_an_error_and_is_not_made_again` |
| B06u | Unix: an operation makes the lock file again if it has gone | KILLED | `a_lock_file_that_has_gone_is_an_error_and_is_not_made_again` |
| B08 | Windows: the lock file is not judged by its names | KILLED | `a_lock_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| B10 | Windows: the open does not make the lock file | KILLED | `a_put_and_a_delete_flush_the_folder`, `a_put_in_progress_survives_another_process_opening_the_store`, `a_put_whose_read_back_fails_is_an_error_and_changes_nothing` |
| Y06 | debris: a dot-file with a longer random part counts as the store's temporary file | KILLED | `a_file_with_a_longer_random_part_than_the_stores_is_not_its_temporary_file` |
| Y07 | marker: an empty provider identifier is a marker | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| Y10 | the identity compares the file index and not the volume | KILLED | `two_files_are_one_only_when_the_volume_and_the_index_are_both_the_same`, `a_path_whose_folder_has_another_volume_or_another_index_than_the_held_one_leads_elsewhere` |
| Y13 | the lock file is not judged at all (a reparse point, a folder, a second name) | KILLED | `a_lock_file_with_another_hard_link_is_refused_until_the_extra_name_is_gone` |
| Y14 | a dangling link under another provider's name does not count as an entry (Windows) | KILLED | `a_link_to_nothing_under_another_providers_extension_is_still_a_value_of_that_provider` |
| Y15 | a file with an invalid stem counts as another provider's secret | KILLED | `a_folder_remembers_its_provider_and_refuses_to_be_opened_with_another` |
| Y16 | a key file is opened following a reparse point (needs SeCreateSymbolicLinkPrivilege to test) | SURVIVED |  |
| Y18 | list: the names are not sorted (Windows: NTFS keeps names sorted) | SURVIVED |  |
| Y18l | list: the names are not sorted (Linux) | KILLED | `list_is_sorted`, `round_trip_overwrite_delete_and_list_for_every_provider` |
| Y51 | a dangling link under another provider's name does not count as an entry (Unix) | KILLED | `a_link_to_nothing_under_another_providers_extension_is_still_a_value_of_that_provider` |
| Y54 | the lock file is opened following a symbolic link (Unix) | KILLED | `a_symbolic_link_in_place_of_the_lock_file_is_refused` |
| Y16w | a key file is opened following a reparse point (the opt-in test that makes the link through WSL) | KILLED | `a_file_reparse_point_made_by_wsl_in_place_of_a_key_file_is_refused_as_a_link_not_opened` |

**The two that survived**, and why they are not a gap:

- **Y16** (a key file is opened following a reparse point) is killed only by an opt-in test: the default suite cannot make a file reparse point on an account that does not hold `SeCreateSymbolicLinkPrivilege` (this machine's), and the test that uses `symlink_file` ends with a message there (and **has not been run** on this machine: it is for the hosted Windows runners). Y16w is the same mutant run against the opt-in test that makes the link through WSL, which needs no privilege (`wsl ln -s` on a Windows drive makes a file reparse point that Win32 cannot open by following it): **killed** (`cargo test -p oaiy-keystore --test keystore -- --ignored wsl`).
- **Y18** (`list` does not sort) is equivalent on NTFS, which keeps the names of a folder in order (and the test that asserts the order passes with or without the sort); Y18l is the same mutant on Linux, where the order is the file system's own: **killed**.

Y12 and Y53 of the review (every failure to flush the folder is success) are A07 and A07l, **killed**; Y10 (the volume is not compared) is killed by a unit test of the comparison with made-up identities and by one that makes the held folder's identity differ in the volume and in the index (there is no second volume to mount in a test without an administrator); Y14 and Y51 (a link to nothing hides another provider's secret) by a test that makes one on each platform (a symbolic link, a junction whose target is gone); Y54 by a symbolic link in place of the lock file on Linux; Y06, Y07 and Y15 by cases added to the tests of the debris shape and of the marker; Y13 by a hard link to the lock file (which Unix refused and Windows did not).