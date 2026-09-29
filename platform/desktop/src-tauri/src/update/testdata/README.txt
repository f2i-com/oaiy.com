A real signature, for the tests.

installer.bin       3000 bytes that are not an installer
installer.bin.sig   its signature, made by `tauri signer sign` (the Tauri CLI this repository uses, 2.11)
throwaway.key.pub   the public key of a THROWAWAY key, made for this and for the pipeline test build.
                    Its private half was not kept. It is not the key OAIY's updates are signed with
                    (that one is in tauri.conf.json, plugins.updater.pubkey).

The Rust tests (update/verify.rs) and the Node tests (platform/scripts/minisign.test.mjs) both verify this
signature: what they accept is what an installed OAIY accepts.
