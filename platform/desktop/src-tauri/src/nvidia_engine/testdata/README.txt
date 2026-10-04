Real signatures, for the tests of nvidia_engine.rs.

engine-windows.zip.bin       a zip holding a stand-in program ("a stand-in for oaiy-llm-server\n") under both
                             platforms' names (oaiy-llm-server.exe and oaiy-llm-server), signed as
                             oaiy-cuda-engine-0.1.0-windows-x64.zip
engine-windows.zip.bin.sig   its signature, made by `tauri signer sign` (the Tauri CLI this repository uses)
engine-linux.tar.gz.bin      the same in a .tar.gz, signed as oaiy-cuda-engine-0.1.0-linux-x86_64.tar.gz
engine-linux.tar.gz.bin.sig  its signature
throwaway.key.pub            the public key of a THROWAWAY key, made for these files. Its private half was
                             deleted the moment they were signed. It is not the key OAIY's releases are signed
                             with (that one is in tauri.conf.json, plugins.updater.pubkey).

The Tauri CLI writes the name of the file it signs into the signature's trusted comment, and the desktop takes an
engine only when that name is the asset it asked for (this version, this platform), so the files were signed under
the asset names and are kept under others.

To make them again: `npx tauri signer generate -w <a folder outside every repository> --ci -p <password>`, put the
two archives under their asset names beside it, put the password in TAURI_SIGNING_PRIVATE_KEY_PASSWORD (not on a
command line) and `npx tauri signer sign -f <the key> <file>` for each, copy the archives and the .sig files here
under the names above with the new .pub, and delete the private key.
