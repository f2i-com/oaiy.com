//! Make a backup of a throwaway data folder with age's default work factor, and restore it onto
//! another, end to end, through the core module (no window, no live data).
//!
//!     cargo run --example backup_demo -- <work dir> <passphrase>
//!
//! It makes `<work dir>/source` (a realistic data folder, with credentials in it), backs it up to
//! `<work dir>/demo.oaiybackup`, then looks at, stages and applies a restore onto
//! `<work dir>/target`, and prints what went where. Only names and counts are printed.

use std::path::{Path, PathBuf};

use oaiy_desktop_lib::backup::create::{create, CreateOptions};
use oaiy_desktop_lib::backup::restore::{self, ApplyOutcome, RestoreOptions};
use oaiy_desktop_lib::backup::review::Ticks;

fn put(root: &Path, rel: &str, body: &[u8]) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

fn tree(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.push(format!("{} ({} bytes)", path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), entry.metadata().unwrap().len()));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let work = PathBuf::from(args.next().expect("a work folder"));
    let passphrase = args.next().expect("a passphrase of at least 12 characters");
    let (source, target, file) = (work.join("source"), work.join("target"), work.join("demo.oaiybackup"));
    for dir in [&source, &target] {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
    }
    let _ = std::fs::remove_file(&file);

    // A used installation: personal data, credentials and heavy things.
    put(&source, "callers.json", b"{\"contacts\":[{\"number\":\"0491 570 006\",\"name\":\"Alex\"}]}");
    put(&source, "callers.json.bak", b"{}");
    put(&source, "calendar/calendar.json", b"{\"appointments\":[]}");
    put(&source, "triggers.json", b"[{\"id\":\"call-in\",\"event\":\"aokie.call.incoming\",\"flowId\":\"greeting\",\"mode\":\"async\"}]");
    put(&source, "flows/greeting.json", b"{\"name\":\"Greeting\",\"nodes\":[{\"id\":\"a\",\"type\":\"logic_block\",\"data\":{}}],\"edges\":[]}");
    put(&source, "setup.json", b"{\"firstRun\":{\"finished\":true}}");
    put(&source, "voices/receptionist.wav", &vec![7u8; 5000]);
    put(&source, "templates/my-rig.json", b"{\"id\":\"my-rig\",\"name\":\"My rig\",\"run\":{\"command\":\"my-rig.exe\",\"args\":[\"--serve\"]},\"autostart\":true}");
    put(&source, "services-autostart.json", b"[\"my-rig\"]");
    put(&source, "plugin-data/aokie/settings.json", b"{\"settings\":{\"greeting\":\"hello, this is the front desk\",\"bargeSensitivity\":100,\"outboundEnabled\":true}}");
    put(&source, "plugin-data/aokie/pairing.json", b"{\"phone\":\"paired\"}");
    put(&source, "ai/providers.json", b"{\"providers\":[{\"apiKey\":\"sk-not-a-real-key\"}]}");
    put(&source, "ai/codex-home/auth.json", b"{\"tokens\":\"x\"}");
    put(&source, "link/account.json", b"{\"credential\":\"flk_not_a_real_key\"}");
    put(&source, "desktop-e2e-identity.key", b"not-a-real-key");
    put(&source, "companion/aokie/endpoint.key", b"not-a-real-key");
    put(&source, "bridge/pairings.json", b"{}");
    put(&source, "models/model.gguf", &vec![1u8; 20_000]);
    put(&source, "logs/oaiy-desktop.log", b"a log line");
    put(&source, "plugins/aokie/manifest.json", b"{}");
    put(&target, "callers.json", b"{\"contacts\":[{\"name\":\"Somebody else\"}]}");
    put(&target, "link/account.json", b"{\"credential\":\"flk_target_own\"}");

    let started = std::time::Instant::now();
    let mut options = CreateOptions::new(&source, &file, &passphrase);
    // (No page to ask for the Agent's storage in this demo.)
    options.agent = None;
    let made = create(&options).expect("the backup is made");
    println!("made {} in {:.1}s: {} bytes, {} files, verified: {}", made.file_name, started.elapsed().as_secs_f32(), made.size, made.counts.files, made.verified);
    println!("warnings: {:?}", made.partial);
    println!("left out on purpose:");
    for e in &made.excluded {
        println!("  {}{}", e.pattern, if e.redo.is_some() { "  (to do again after a restore)" } else { "" });
    }

    let opts = RestoreOptions::default();
    let started = std::time::Instant::now();
    let preview = restore::inspect(&target, &file, &passphrase, &opts).expect("the dry run");
    println!("\ndry run in {:.1}s (nothing changed):", started.elapsed().as_secs_f32());
    for c in &preview.categories {
        println!("  {:<48} added {} replaced {} unchanged {} left alone {}", c.label, c.added, c.replaced, c.unchanged, c.left_alone);
    }
    println!("  the backup lacks: {:?}", preview.lacks);
    println!("  to do again: {:?}", preview.redo);
    println!("  what can act, and needs a tick ({} kinds, {} items):", preview.classes.len(), preview.items.len());
    for c in &preview.classes {
        println!("    [{}] {} ({} item(s))", c.id, c.label, c.count);
    }
    for item in &preview.items {
        println!("    {} :: {} :: {}", item.name, item.title, item.what);
    }

    // Round 1: nothing ticked, as it is to begin with. Only data comes back.
    let staged = restore::stage(&target, &file, &passphrase, &Ticks::none(), &opts).expect("staged");
    println!("\nstaged with nothing ticked: {} files ({} bytes); left out: {:?}", staged.files, staged.bytes, staged.skipped);
    match restore::apply_pending(&target) {
        ApplyOutcome::Applied(done) => println!("applied at the next start: ok = {}", done.ok),
        other => panic!("not applied: {other:?}"),
    }
    println!("the target folder now holds (no template, no autostart list, no settings):");
    for line in tree(&target) {
        println!("  {line}");
    }
    assert!(!target.join("templates").exists() && !target.join("services-autostart.json").exists(), "nothing that can run came back unticked");

    // Round 2: everything ticked (your own backup, an explicit click).
    let staged = restore::stage(&target, &file, &passphrase, &Ticks::all(), &opts).expect("staged");
    println!("\nstaged with every kind ticked: {} files ({} bytes)", staged.files, staged.bytes);
    println!("applying at the next start:");
    match restore::apply_pending(&target) {
        ApplyOutcome::Applied(done) => println!("  applied: ok = {}", done.ok),
        other => panic!("not applied: {other:?}"),
    }
    println!("the target folder now holds:");
    for line in tree(&target) {
        println!("  {line}");
    }
    println!("the credential that was already there is unchanged: {}", std::fs::read_to_string(target.join("link/account.json")).unwrap().contains("flk_target_own"));
    match restore::apply_pending(&target) {
        ApplyOutcome::None => println!("a second start applies nothing"),
        other => panic!("applied twice: {other:?}"),
    }
}
