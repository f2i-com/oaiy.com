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
    // The engines are asked afresh when an install is decided, and a studio that cannot say is not taken for idle.
    assert!(probes.contains("crate::engines::activity(fresh)"), "the engines are no longer asked with the fresh flag an install sets");
    assert!(probes.contains("Err(why) => crate::update::blockers::EnginesState::Unknown(why)"), "engines that cannot say are taken for something else than unknown");
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

#[test]
fn the_hand_off_looks_at_the_package_before_the_installer_is_given_anything() {
    let gui = source(include_str!("gui.rs"));
    let hand_off = block(&gui, "let hand_off = |package: &VerifiedPackage| {", "\n    };");
    let look = hand_off.find("package.check_for_hand_off(&update.version, Target::current())?;").expect("the hand-off no longer checks the package against the update handle and this platform");
    let install = hand_off.find("update.install(package.bytes())").expect("the hand-off no longer installs the package's bytes");
    assert!(look < install, "the package is looked at after the installer has it");
}

#[test]
fn the_address_of_the_plugins_update_is_checked_for_this_platforms_installer() {
    // prepare() works on the plugin's Update handle too. The address it names must be this platform's installer by its exact
    // name (feed::check_asset_url), whatever else of the release the feed points at.
    let gui = source(include_str!("gui.rs"));
    let prepare = block(&gui, "fn prepare<'a>(", "\n    }\n");
    assert!(prepare.contains("Target::current()"), "prepare no longer asks which kind of installer this platform takes");
    assert!(prepare.contains("check_asset_url(update.download_url.as_str(), &update.version, target)?;"), "the update's address is no longer checked for this platform and version");
}

#[test]
fn a_panic_of_the_install_task_is_turned_into_a_failed_update() {
    let gui = source(include_str!("gui.rs"));
    let command = block(&gui, "pub async fn update_install(", "\n}\n");
    assert!(command.contains("fail_install_if_installing("), "the install task's panic would leave the update on installing");
    let install = source(include_str!("install.rs"));
    let perform = block(&install, "pub fn perform(", "\n}\n");
    assert!(perform.contains("Unwind {"), "the sequence no longer holds the guard that undoes a panic");
}

#[test]
fn the_plugins_part_says_it_holds_the_calls_so_they_are_looked_at_once_more_right_before_it_stops() {
    let gui = source(include_str!("gui.rs"));
    let plugins = block(&gui, "impl Part for PluginsPart {", "\n}\n");
    assert!(plugins.contains("fn holds_calls(&self) -> bool {\n        true\n    }"), "stopping the plugins ends a phone call, and the last look for one is no longer taken before it");
    let install = source(include_str!("install.rs"));
    let perform = block(&install, "pub fn perform(", "\n}\n");
    let look = perform.find("updater.call_blockers()").expect("perform no longer looks at the calls before a part that holds them");
    let stop = perform.find("part.stop()").expect("perform no longer stops parts");
    assert!(look < stop, "the look for calls comes after the stop it is for");
}

/// Every Rust file under src/ (but this one, which names what it looks for), with its comments left out: `(path from src/, code)`.
fn code_of_every_source_file() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("src/ is readable").flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    files
        .into_iter()
        .map(|path| (path.strip_prefix(&src).unwrap().to_string_lossy().replace('\\', "/"), path))
        .filter(|(name, _)| name != "update/guards.rs")
        .map(|(name, path)| {
            let text = source(&std::fs::read_to_string(&path).expect("a source file"));
            // A line comment is not code: drop it, unless the // is inside a string.
            let code: Vec<String> = text
                .lines()
                .map(|line| match line.find("//") {
                    Some(at) if line[..at].matches('"').count() % 2 == 0 => line[..at].to_string(),
                    _ => line.to_string(),
                })
                .collect();
            (name, code.join("\n"))
        })
        .collect()
}

#[test]
fn nothing_installs_or_downloads_an_update_but_the_two_calls_that_are_held_to_a_verified_package() {
    let files = code_of_every_source_file();
    assert!(files.len() > 50, "the source tree was not found: {} files", files.len());
    // The plugin's download-and-install in one go would skip the check of the bytes and of the moment: it is named nowhere.
    for (name, code) in &files {
        assert!(!code.contains("download_and_install"), "{name} names the plugin's download_and_install");
    }
    // The plugin's crate is used by one file, and it is this one's sibling that holds the calls.
    for (name, code) in &files {
        if name != "update/gui.rs" {
            assert!(!code.contains("tauri_plugin_updater") && !code.contains("UpdaterExt"), "{name} uses the updater plugin: only update/gui.rs may");
        }
    }
    // An install with bytes is called from one place, and what it is given is a verified package's bytes.
    let installs: Vec<(&str, usize)> = files.iter().map(|(name, code)| (name.as_str(), code.matches(".install(").count() - code.matches(".install()").count())).filter(|(_, n)| *n > 0).collect();
    assert_eq!(installs, [("update/gui.rs", 1)], "an install that takes bytes is called from one place only");
    let gui = &files.iter().find(|(name, _)| name == "update/gui.rs").unwrap().1;
    assert_eq!(gui.matches("update.install(package.bytes())").count(), 1, "the one install is not given a verified package's bytes");
    // The download is called from one place, and its bytes go to verify_package before anything else has them.
    let downloads: Vec<(&str, usize)> = files.iter().map(|(name, code)| (name.as_str(), code.matches(".download(").count())).filter(|(_, n)| *n > 0).collect();
    assert_eq!(downloads, [("update/gui.rs", 1)], "an update is downloaded from one place only");
    let download = block(gui, "async fn download(", "\n}\n");
    assert!(download.find("update.download(").unwrap() < download.find("verify_package(bytes,").expect("the download is verified"), "the bytes are verified after the download and before they go anywhere");
}
