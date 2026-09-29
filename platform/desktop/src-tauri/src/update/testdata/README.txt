Real signatures, for the tests.

windows-setup.bin       3000 bytes that start like a Windows executable (MZ) and are not an installer,
                        signed as OAIY_0.1.0_x64-setup.exe
windows-setup.bin.sig   its signature, made by `tauri signer sign` (the Tauri CLI this repository uses, 2.11)
linux-appimage.bin      3000 bytes that start like a Linux executable (ELF) and are not an AppImage,
                        signed as OAIY_0.1.0_amd64.AppImage
linux-appimage.bin.sig  its signature
throwaway.key.pub       the public key of a THROWAWAY key, made for these files. Its private half was
                        deleted the moment they were signed. It is not the key OAIY's updates are signed
                        with (that one is in tauri.conf.json, plugins.updater.pubkey).

The files are named as the bundler names an installer when they are signed, because the Tauri CLI
writes the name of the file it signs into the signature's trusted comment (`file:OAIY_0.1.0_x64-setup.exe`)
and the desktop holds a signature to the version it is announced as by that name. The bytes are stored
under other names (a file named .exe that is not a program is better not kept), and the signature covers
bytes, not the name of the file they are kept in.

The Rust tests (update/verify.rs, update/target.rs) and the Node tests (platform/scripts/minisign.test.mjs,
make-latest-json.test.mjs) verify these signatures: what they accept is what an installed OAIY accepts.

To make them again: `npx tauri signer generate -w <a folder outside every repository>`, put the two payloads under
their bundler names beside it, put the password in TAURI_SIGNING_PRIVATE_KEY_PASSWORD (not on a command line)
and `npx tauri signer sign -f <the key> <file>` for each, copy the payloads and the .sig files here under the names above with the new .pub, and delete the private key.

aokie-manifest.json     a COPY of the manifest of the phone plugin OAIY is used with (aokie.com,
                        crates/aokie-plugin/manifest.json, at cb298dd). update/phone.rs runs it through the
                        plugin gate to show that the two commands an update asks whether a call is live
                        (call.switchboard, call.current) are declared and not journalled, so they need no key.
                        Copy it again when that plugin's commands change.
aokie-phone-definition.json   the service definition that manifest requires (definitions/phone.json), likewise a copy.
