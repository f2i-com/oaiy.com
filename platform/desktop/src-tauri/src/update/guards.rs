//! Tests that read the source itself, for what a unit test cannot run: the desktop's start-up code
//! (a whole Tauri app) and the rule that only one place may install anything.
//!
//! They are blunt on purpose: each fails when a line that a safety property rests on is removed or
//! moved, and says which.

/// A source file as the tests read it, with the line endings of a Windows checkout made plain.
fn source(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// The text from `start` up to the first line after it that is just `close` (with its indent), or panics.
fn block<'a>(text: &'a str, start: &str, close: &str) -> &'a str {
    let from = text.find(start).unwrap_or_else(|| panic!("`{start}` is gone from the source"));
    let rest = &text[from..];
    let to = rest.find(close).unwrap_or_else(|| panic!("`{close}` does not follow `{start}`"));
    &rest[..to]
}

#[test]
fn the_desktops_activity_probes_ask_the_phone_plugin_the_engines_and_the_rest() {
    // The desktop builds its Probes in one place (lib.rs: `run`), inside an async block that cannot be run in a unit
    // test. If a field is dropped there the update goes on without asking that source, and nothing else notices.
    let lib = source(include_str!("../lib.rs"));
    let probes = block(&lib, "crate::update::blockers::Probes {", "\n                    },\n                );");
    for field in ["downloads: Some(", "registry: Some(", "python: Some(", "node: Some(", "engines: Some(", "migration: Some(", "phone: Some("] {
        assert!(probes.contains(field), "the desktop's update Probes no longer has `{field}`: an install would not ask that source\n{probes}");
    }
    assert!(probes.contains("crate::update::phone::PluginLine::new("), "the phone source is not the plugin line");
    assert!(!probes.contains("..Default::default()"), "a field left to the default is a source not asked");
}

#[test]
fn the_download_holds_its_signature_to_the_announced_version_and_this_platforms_installer() {
    // The download itself is a function of the plugin's Update handle, which a unit test cannot make. What it must do before it
    // hands back a package is what verify.rs tests: ask for THIS platform's installer, for the version the update was found as.
    let gui = source(include_str!("gui.rs"));
    let download = block(&gui, "async fn download(", "\n}\n");
    assert!(download.contains("Target::current()"), "the download no longer asks which kind of installer this platform takes");
    assert!(download.contains("let (version, signature) = (update.version.clone(), update.signature.clone());"), "the version is no longer the update handle's own");
    assert!(download.contains("Expected::new(&version, target)"), "the signature is no longer held to the announced version and this platform's installer");
    assert!(download.contains("verify_package(bytes, &signature, &pubkey, &expected)"), "what the plugin fetched no longer goes through verify_package");
}
