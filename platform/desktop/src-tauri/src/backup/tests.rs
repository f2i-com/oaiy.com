//! Tests for the backup: what is included and left out, the file, restoring, and what must never leak.
//!
//! They work on throwaway data folders under the system temp folder (never the live one) with
//! realistic files, use a small scrypt work factor so each file opens in milliseconds, and check the
//! format with the age crate directly, apart from the code under test.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Cursor, Read, Write};
use std::path::Path;
use std::sync::{Mutex, Once};

use sha2::{Digest, Sha256};

use super::agent::{self, AgentExport, DonePayload, PartError, MISSING_WARNING, PART_SIZE};
use super::busy::Busy;
use crate::update::blockers::{self, fake::Fake, Activity, EnginesState, Readings};
use crate::update::phone::LineState;
use super::container::{self, Cost};
use super::create::{self, create, CreateOptions, CreateResult};
use super::manifest::{AppInfo, Counts, Entry, Manifest};
use super::restore::{self, ApplyOutcome, Inject, RestoreOptions};
use super::review::{RestoreClass, Ticks};
use super::rules::{self, Category, Decision};
use super::*;
use crate::secret_file::testing::{assert_private, TempDir};

const PASS: &str = "correct horse battery staple";
const CANARY_PASS: &str = "PASSPHRASE-CANARY-9f3a71";
const LINK_KEY: &str = "flk_CANARY_link_credential_0001";
const PROVIDER_KEY: &str = "sk-CANARY-provider-key-0002";

/// Backups are made one at a time in the app; the tests share the process, so they take turns.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn put(root: &Path, rel: &str, body: impl AsRef<[u8]>) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn get(root: &Path, rel: &str) -> Option<Vec<u8>> {
    fs::read(root.join(rel)).ok()
}

/// A JSON file's content, so cleaned files (pretty-printed on the way) compare by what they say.
fn json_of(root: &Path, rel: &str) -> serde_json::Value {
    serde_json::from_slice(&fs::read(root.join(rel)).unwrap_or_else(|_| panic!("{rel} exists"))).unwrap_or_else(|_| panic!("{rel} is JSON"))
}

/// Every file under `root` (relative name and bytes), except the backup's and restore's own folders.
fn snapshot(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            if rel == "restore" || rel == "backup" {
                continue;
            }
            let meta = fs::symlink_metadata(&path).unwrap();
            if rules::is_link(&meta) {
                out.insert(format!("{rel} (link)"), Vec::new());
            } else if meta.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(rel, fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// A data folder as a used installation has it: personal data, credentials and everything heavy.
fn realistic(root: &Path, tag: &str) {
    put(root, "callers.json", format!("{{\"contacts\":[{{\"number\":\"0491 570 006\",\"name\":\"Alex ({tag})\",\"facts\":[\"likes email\"]}}]}}"));
    put(root, "callers.json.bak", b"{\"older\":true}");
    put(root, "calendar/calendar.json", format!("{{\"appointments\":[{{\"id\":\"a1\",\"title\":\"Check-up ({tag})\"}}]}}"));
    put(root, "triggers.json", format!("[{{\"id\":\"t1\",\"event\":\"aokie.call.incoming\",\"flowId\":\"greeting\",\"mode\":\"async\",\"inputMap\":{{\"note\":\"{tag}\"}}}}]"));
    put(root, "flows/greeting.json", format!("{{\"name\":\"Greeting\",\"tag\":\"{tag}\"}}"));
    put(root, "flows/token-refund.json", format!("{{\"name\":\"Refund token flow\",\"tag\":\"{tag}\"}}"));
    put(root, "setup.json", format!("{{\"firstRun\":{{\"finished\":true}},\"tag\":\"{tag}\"}}"));
    put(root, "agent.json", b"{\"model\":\"engine\"}");
    put(root, "control.json", b"{\"agentMayChange\":true}");
    put(root, "services-autostart.json", b"[\"oaiy-voice\"]");
    put(root, "connectors/formlogic.json", b"{\"id\":\"formlogic\"}");
    put(root, "control-log.jsonl", b"{\"tool\":\"x\"}\n");
    put(root, "bridge/ledger.jsonl", b"{\"run\":1}\n");
    put(root, "bridge/deadletters.jsonl", b"");
    put(root, "voices/receptionist.wav", vec![7u8; 4000]);
    put(root, "voices/chosen", b"front-desk");
    put(root, "templates/my-rig.json", b"{\"id\":\"my-rig\"}");
    put(root, "templates/ollama.json", b"{\"id\":\"ollama\"}");
    put(root, "templates/.ollama.json.seed", b"{\"id\":\"ollama\"}");
    put(root, "templates/edited.json", b"{\"id\":\"edited\",\"mine\":true}");
    put(root, "templates/.edited.json.seed", b"{\"id\":\"edited\"}");
    put(root, "plugin-data/aokie/settings.json", format!("{{\"greeting\":\"hello ({tag})\"}}"));
    put(root, "plugin-data/aokie/notes.txt", b"the plugin's notes");
    put(root, "plugin-data/aokie/pairing.json", b"{\"phone\":\"secret-pairing\"}");
    put(root, "plugin-data/aokie/outbox.db", b"sealed rows");
    put(root, "ai/providers.json", format!("{{\"providers\":[{{\"id\":\"p1\",\"apiKey\":\"{PROVIDER_KEY}\"}}]}}"));
    put(root, "ai/codex-home/auth.json", b"{\"tokens\":\"chatgpt\"}");
    put(root, "link/account.json", format!("{{\"credential\":\"{LINK_KEY}\",\"instance\":\"i-1\"}}"));
    put(root, "link/outbox/1.json", b"{\"event\":1}");
    put(root, "link/app-logic-storage.json", b"{}");
    put(root, "desktop-e2e-identity.key", b"AAAAtunnel-identity-key");
    put(root, "data-node-signing.key", b"BBBBdata-node-key");
    put(root, "companion/aokie/endpoint.key", b"CCCCendpoint");
    put(root, "companion/aokie/roster.json", b"{\"phones\":[]}");
    put(root, "companion/relay.json", b"{\"bearer\":\"relay\"}");
    put(root, "companion/upstream.json", b"{\"bearer\":\"issuer\"}");
    put(root, "bridge/pairings.json", b"{\"tokens\":[\"paired\"]}");
    put(root, "engines/oaiy-studio.json", b"{\"gateway_key\":\"g\",\"hf_token\":\"hf_x\"}");
    put(root, "models/model.gguf", vec![1u8; 20_000]);
    put(root, "python/python.exe", b"py");
    put(root, "venvs/a/pyvenv.cfg", b"cfg");
    put(root, "node/node.exe", b"node");
    put(root, "bin/tool.exe", b"tool");
    put(root, "logs/oaiy-desktop.log", b"a log line");
    put(root, "tmp/scratch", b"tmp");
    put(root, "plugins/aokie/manifest.json", b"{}");
    put(root, "plugins/.backup-aokie-1f/manifest.json", b"{}");
    put(root, "plugins/trusted-plugins.json", b"{}");
    put(root, "scripts/install.ps1", b"echo");
    put(root, "model-catalog.json", b"[]");
    put(root, "services-running.json", b"[]");
    put(root, "mystery.bin", b"who knows");
    put(root, "unknown-dir/file.txt", b"?");
}

/// What another computer might already have, to restore onto.
fn target(root: &Path) {
    put(root, "callers.json", b"{\"contacts\":[{\"name\":\"Somebody else\"}]}");
    put(root, "flows/local-only.json", b"{\"name\":\"only here\"}");
    put(root, "flows/greeting.json", b"{\"name\":\"Greeting\",\"tag\":\"old\"}");
    put(root, "link/account.json", b"{\"credential\":\"flk_TARGET_OWN\"}");
    put(root, "ai/providers.json", b"{\"providers\":[{\"apiKey\":\"sk-TARGET-OWN\"}]}");
    put(root, "companion/aokie/endpoint.key", b"TARGET-endpoint");
    put(root, "models/keep.gguf", b"keep");
}

fn options() -> RestoreOptions {
    RestoreOptions::default()
}

fn make_with(data: &Path, dest: &Path, pass: &str, keys: bool, agent: Option<&dyn AgentExport>) -> Result<CreateResult> {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut o = CreateOptions::new(data, dest, pass);
    o.include_keys = keys;
    o.cost = Cost::Fixed(8);
    o.agent = agent;
    o.agent_wait = create::short_wait();
    create(&o)
}

fn make(data: &Path, dest: &Path) -> CreateResult {
    make_with(data, dest, PASS, false, None).expect("the backup is made")
}

fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The decrypted ZIP of a backup, read with the age crate directly.
fn plain_zip(file: &Path, pass: &str) -> Vec<u8> {
    let decryptor = age::Decryptor::new_buffered(BufReader::new(File::open(file).unwrap())).unwrap();
    assert!(decryptor.is_scrypt());
    let identity = age::scrypt::Identity::new(age::secrecy::SecretString::from(pass.to_string()));
    let mut reader = decryptor.decrypt(std::iter::once(&identity as &dyn age::Identity)).unwrap();
    let mut out = Vec::new();
    reader.read_to_end(&mut out).unwrap();
    out
}

fn manifest_of(file: &Path, pass: &str) -> Manifest {
    let zip = plain_zip(file, pass);
    let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
    let mut first = archive.by_index(0).unwrap();
    assert_eq!(first.name(), "manifest.json", "the manifest is the first entry");
    let mut text = String::new();
    first.read_to_string(&mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Build an encrypted backup by hand, for the tests that need one no honest writer makes.
fn craft(dest: &Path, manifest: &Manifest, files: &[(&str, &[u8])], manifest_first: bool) {
    let zip_path = dest.with_extension("zip");
    let mut writer = zip::ZipWriter::new(File::create(&zip_path).unwrap());
    let opts = zip::write::SimpleFileOptions::default();
    if manifest_first {
        writer.start_file("manifest.json", opts).unwrap();
        writer.write_all(&manifest.to_json()).unwrap();
    }
    for (name, bytes) in files {
        writer.start_file(*name, opts).unwrap();
        writer.write_all(bytes).unwrap();
    }
    if !manifest_first {
        writer.start_file("manifest.json", opts).unwrap();
        writer.write_all(&manifest.to_json()).unwrap();
    }
    writer.finish().unwrap();
    let _ = fs::remove_file(dest);
    container::encrypt_file(&zip_path, dest, PASS, Cost::Fixed(8)).unwrap();
    let _ = fs::remove_file(&zip_path);
}

fn manifest_for(files: &[(&str, &[u8])]) -> Manifest {
    Manifest {
        v: 1,
        created_at: "2026-09-30T01:02:03Z".into(),
        app: AppInfo { name: "oaiy".into(), version: "0.1.0".into() },
        platform: "windows".into(),
        entries: files.iter().map(|(n, b)| Entry { name: n.to_string(), size: b.len() as u64, sha256: sha(b) }).collect(),
        excluded: Vec::new(),
        counts: Counts::default(),
        includes_keys: false,
        partial: Vec::new(),
    }
}

/// Nothing was staged and nothing waits.
fn assert_nothing_staged(data: &Path) {
    let restore = data.join("restore");
    assert!(!restore.join("pending.json").exists(), "no marker");
    let leftovers: Vec<_> = fs::read_dir(&restore).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    assert!(leftovers.iter().all(|n| !n.starts_with("pending-")), "nothing staged: {leftovers:?}");
    let scratch = data.join("backup").join("scratch");
    let held: Vec<_> = fs::read_dir(&scratch).map(|d| d.flatten().collect::<Vec<_>>()).unwrap_or_default();
    assert!(held.is_empty(), "no scratch plaintext is left behind");
}

// ---- what a backup holds and leaves out ------------------------------------------------------------

/// One test per item a backup leaves out: with only that file in the folder, it is not in the plan,
/// it is listed with a reason, and a credential says what to do again.
macro_rules! left_out {
    ($name:ident, $path:expr, $pattern:expr, $redo:expr) => {
        #[test]
        fn $name() {
            let dir = TempDir::new("left-out");
            put(&dir.0, "callers.json", b"{}");
            put(&dir.0, $path, b"contents that must not be copied");
            let plan = rules::plan(&dir.0, false);
            let names: Vec<&str> = plan.items.iter().map(|i| i.rel.as_str()).collect();
            assert_eq!(names, ["callers.json"], "{} is not backed up", $path);
            let record = plan.excluded.iter().find(|e| e.pattern.contains($pattern)).unwrap_or_else(|| panic!("{} is listed as left out: {:?}", $path, plan.excluded));
            assert!(record.reason.len() > 12, "it says why");
            assert_eq!(record.redo.is_some(), $redo, "what to do again after a restore");
            assert!(matches!(rules::classify($path, false), Decision::Exclude(_)), "and a restore refuses it");
            if $path != "ai/providers.json" {
                assert!(rules::category_of_backup_entry($path).is_err());
            }
        }
    };
}

left_out!(models_are_left_out, "models/model.gguf", "models/**", false);
left_out!(engines_are_left_out, "engines/oaiy-llm-server.exe", "engines/**", true);
left_out!(the_engines_settings_with_their_keys_are_left_out, "engines/oaiy-studio.json", "engines/**", true);
left_out!(python_is_left_out, "python/python.exe", "python/**", false);
left_out!(venvs_are_left_out, "venvs/a/pyvenv.cfg", "venvs/**", false);
left_out!(node_is_left_out, "node/node.exe", "node/**", false);
left_out!(bin_is_left_out, "bin/tool.exe", "bin/**", false);
left_out!(logs_are_left_out, "logs/oaiy-desktop.log", "logs/**", false);
left_out!(caches_are_left_out, "cache/x.bin", "cache/**", false);
left_out!(temp_files_are_left_out, "tmp/scratch", "tmp/**", false);
left_out!(bak_copies_are_left_out, "callers.json.bak", "*.bak", false);
left_out!(half_written_files_are_left_out, "flows/x.json.tmp", "*.tmp", false);
left_out!(corrupt_copies_are_left_out, "ai/providers.json.corrupt", "*.corrupt", false);
left_out!(plugin_rollback_folders_are_left_out, "plugins/.backup-aokie-1f/manifest.json", "plugins/**", true);
left_out!(the_formlogic_link_credential_and_instance_id_are_left_out, "link/account.json", "link/account.json", true);
left_out!(the_links_events_and_markers_are_left_out, "link/outbox/1.json", "link/**", false);
left_out!(the_app_logic_markers_are_left_out, "link/app-logic-storage.json", "link/**", false);
left_out!(the_tunnel_identity_key_is_left_out, "desktop-e2e-identity.key", "desktop-e2e-identity.key", true);
left_out!(the_tunnel_published_marker_is_left_out, "desktop-e2e-published.json", "desktop-e2e-identity.key", true);
left_out!(the_data_node_key_is_left_out, "data-node-signing.key", "data-node-signing.key", true);
left_out!(companion_endpoint_keys_are_left_out, "companion/aokie/endpoint.key", "companion/**", true);
left_out!(the_companion_roster_is_left_out, "companion/aokie/roster.json", "companion/**", true);
left_out!(the_relay_bearer_is_left_out, "companion/relay.json", "companion/**", true);
left_out!(the_issuer_bearer_is_left_out, "companion/upstream.json", "companion/**", true);
left_out!(paired_browser_tokens_are_left_out, "bridge/pairings.json", "bridge/pairings.json", true);
left_out!(the_chatgpt_sign_in_is_left_out, "ai/codex-home/auth.json", "ai/codex-home/**", true);
left_out!(the_hugging_face_token_is_left_out, "hf-token", "hf-token", true);
left_out!(provider_api_keys_are_left_out_by_default, "ai/providers.json", "ai/providers.json", true);
left_out!(key_folders_are_left_out, "keys/ldk.v1.key", "keys/**", false);
left_out!(vault_files_are_left_out, "vault/umk", "keys/**", false);
left_out!(a_plugins_pairing_state_is_left_out, "plugin-data/aokie/pairing.json", "plugin data", true);
left_out!(a_plugins_outbox_is_left_out, "plugin-data/aokie/outbox.db", "plugin data", true);
left_out!(a_machine_bound_blob_is_left_out, "plugin-data/aokie/link.dpapi", "*.dpapi", true);
left_out!(a_sealed_pin_is_left_out, "plugin-data/aokie/manager-pin.sealed", "*.dpapi", true);
left_out!(a_plugins_sign_in_file_is_left_out, "plugin-data/aokie/auth.json", "auth.json", true);
left_out!(a_stray_key_file_is_left_out, "flows/backup.pem", "*.pem", true);
left_out!(installed_plugins_are_left_out, "plugins/aokie/manifest.json", "plugins/**", true);
left_out!(trust_decisions_are_left_out, "plugins/trusted-plugins.json", "plugins/**", true);
left_out!(scripts_are_left_out, "scripts/install.ps1", "scripts/**", false);
left_out!(the_model_catalog_is_left_out, "model-catalog.json", "model-catalog.json", false);
left_out!(seed_snapshots_are_left_out, "templates/.ollama.json.seed", ".*.seed", false);
left_out!(the_backups_own_folders_are_left_out, "backup/status.json", "restore/**, backup/**", false);
left_out!(the_restores_own_folders_are_left_out, "restore/pending.json", "restore/**, backup/**", false);

#[test]
fn personal_data_is_kept_under_its_category() {
    let table: &[(&str, Category)] = &[
        ("callers.json", Category::Contacts),
        ("calendar/calendar.json", Category::Calendar),
        ("triggers.json", Category::Flows),
        ("flows/greeting.json", Category::Flows),
        ("setup.json", Category::Settings),
        ("agent.json", Category::Settings),
        ("control.json", Category::Settings),
        ("services-autostart.json", Category::Templates),
        ("connectors/formlogic.json", Category::Connectors),
        ("control-log.jsonl", Category::History),
        ("bridge/ledger.jsonl", Category::Flows),
        ("bridge/deadletters.jsonl", Category::History),
        ("voices/receptionist.wav", Category::Voices),
        ("templates/my-rig.json", Category::Templates),
        ("plugin-data/aokie/settings.json", Category::PluginData),
    ];
    for (path, category) in table {
        match rules::classify(path, false) {
            Decision::Include { category: got, secret } => {
                assert_eq!(got, *category, "{path}");
                assert!(!secret, "{path} is not a secret");
            }
            _ => panic!("{path} should be kept"),
        }
    }
    // A plugin's data is opt-in, plugin by plugin and file by file.
    for not_kept in ["plugin-data/aokie/notes.txt", "plugin-data/aokie/sub/deeper.json", "plugin-data/aokie/manager-auth.json", "plugin-data/aokie/aokie_radio/pairing_store.json", "plugin-data/unknown/settings.json", "calendar/other.json", "voices/run.exe", "voices/deeper/a.wav"] {
        assert!(!matches!(rules::classify(not_kept, false), Decision::Include { .. }), "{not_kept} must not be kept");
    }
    // A flow named for a token is the person's own flow, not a key.
    assert!(matches!(rules::classify("flows/token-refund.json", false), Decision::Include { .. }));
}

#[test]
fn provider_keys_are_added_only_when_asked() {
    let dir = TempDir::new("keys");
    put(&dir.0, "ai/providers.json", b"{}");
    assert!(rules::plan(&dir.0, false).items.is_empty());
    let with = rules::plan(&dir.0, true);
    assert_eq!(with.items.len(), 1);
    assert_eq!(with.items[0].category, Category::Providers);
    assert!(with.items[0].secret, "it is written private when restored");
    assert!(with.excluded.is_empty());
    // What a backup says about itself decides nothing: a restore takes the provider list as a class the
    // person ticks, and whether its keys come back is a tick of its own.
    assert!(rules::category_of_backup_entry("ai/providers.json").is_ok());
}

#[test]
fn unedited_templates_are_left_out_and_edited_ones_kept() {
    let dir = TempDir::new("templates");
    realistic(&dir.0, "A");
    let plan = rules::plan(&dir.0, false);
    let names: Vec<&str> = plan.items.iter().filter(|i| i.category == Category::Templates && i.rel.starts_with("templates/")).map(|i| i.rel.as_str()).collect();
    assert_eq!(names, ["templates/edited.json", "templates/my-rig.json"]);
    assert!(plan.excluded.iter().any(|e| e.pattern.contains("<built-in>")));
}

#[test]
fn what_is_not_recognised_is_left_out_and_listed() {
    let dir = TempDir::new("unknown");
    realistic(&dir.0, "A");
    let plan = rules::plan(&dir.0, false);
    assert!(plan.items.iter().all(|i| i.rel != "mystery.bin" && !i.rel.starts_with("unknown-dir")));
    assert!(plan.excluded.iter().any(|e| e.pattern == "mystery.bin" && e.reason.contains("Not recognised")));
    assert!(plan.excluded.iter().any(|e| e.pattern == "unknown-dir/"));
}

#[test]
fn a_walk_skips_a_models_folder_without_going_into_it() {
    let dir = TempDir::new("walk");
    put(&dir.0, "callers.json", b"{}");
    put(&dir.0, "models/a/b/c/model.gguf", vec![0u8; 100]);
    let plan = rules::plan(&dir.0, false);
    assert_eq!(plan.items.len(), 1);
    assert!(plan.excluded.iter().any(|e| e.pattern.contains("models/**")));
}

#[test]
fn links_and_junctions_are_never_followed_and_are_listed() {
    let outside = TempDir::new("outside");
    put(&outside.0, "secret-personal.json", b"must never be copied");
    let dir = TempDir::new("links");
    put(&dir.0, "callers.json", b"{}");
    fs::create_dir_all(dir.0.join("voices")).unwrap();
    let Some(_link) = crate::plugins::trust::tests::dir_link(&dir.0.join("voices").join("linked"), &outside.0) else {
        eprintln!("no directory link could be made here: nothing to test");
        return;
    };
    let plan = rules::plan(&dir.0, false);
    assert_eq!(plan.links, ["voices/linked"], "the link is listed");
    assert!(plan.items.iter().all(|i| !i.rel.contains("secret-personal")), "and what it points at is not read");
    // A backup of it records the link in what was left out.
    let out = TempDir::new("links-out");
    let made = make(&dir.0, &out.0.join("b.oaiybackup"));
    assert!(made.excluded.iter().any(|e| e.pattern == "voices/linked" && e.reason.contains("never followed")));
    let zip = plain_zip(&out.0.join("b.oaiybackup"), PASS);
    assert!(!String::from_utf8_lossy(&zip).contains("must never be copied"));
}

// ---- names -----------------------------------------------------------------------------------------

#[test]
fn unsafe_names_are_refused_and_plain_ones_pass() {
    let limits = Limits::default();
    let bad = [
        "", "..", "../x.json", "a/../b.json", "a/./b", "/abs.json", "\\abs.json", "a\\b.json", "C:/x.json", "C:x.json", "a:b", "a//b", "a/", "x\0y", "con.txt",
        "a/nul", "a/COM1.txt", "trailing.", "space ", "a/b /c", "star*", "q?", "pipe|", "quote\"", "ctl\u{1}x", &"x".repeat(513),
    ];
    for name in bad {
        assert!(container::check_entry_name(name, &limits).is_err(), "{name:?} must be refused");
    }
    for name in ["callers.json", "plugin-data/aokie/settings.json", "voices/Ünï.wav", "a/b/c.d.e", "templates/.x.seed"] {
        assert!(container::check_entry_name(name, &limits).is_ok(), "{name:?} is a plain name");
    }
    let root = Path::new("root");
    let joined = container::safe_join(root, "a/b.json", &limits).unwrap();
    assert!(joined.starts_with(root));
    assert!(container::safe_join(root, "../b.json", &limits).is_err());
}

// ---- the file --------------------------------------------------------------------------------------

#[test]
fn the_file_is_a_standard_age_file_around_a_zip_that_starts_with_the_manifest() {
    let data = TempDir::new("format");
    realistic(&data.0, "A");
    let out = TempDir::new("format-out");
    let file = out.0.join("backup.oaiybackup");
    make(&data.0, &file);
    let bytes = fs::read(&file).unwrap();
    assert!(bytes.starts_with(b"age-encryption.org/v1\n-> scrypt "), "the age header, in binary (no armor)");
    assert!(!bytes.windows(5).any(|w| w == b"BEGIN"), "no armor");
    // Opened by the age crate directly, not by this module's code.
    let zip = plain_zip(&file, PASS);
    assert!(zip.starts_with(b"PK"), "a ZIP");
    let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
    assert_eq!(archive.by_index(0).unwrap().name(), "manifest.json");
    // Every name in it is relative with forward slashes.
    for i in 0..archive.len() {
        let name = archive.by_index(i).unwrap().name().to_string();
        assert!(!name.contains('\\') && !name.starts_with('/'), "{name}");
    }
    // Nothing readable in the file itself.
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains("Alex") && !text.contains("callers.json") && !text.contains(PROVIDER_KEY));
    // The final name is the only file made in the folder.
    let names: Vec<String> = fs::read_dir(&out.0).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["backup.oaiybackup"], "no .tmp is left");
    assert_private(&file);
}

#[test]
fn the_manifest_records_what_the_brief_says() {
    let data = TempDir::new("manifest");
    realistic(&data.0, "A");
    let out = TempDir::new("manifest-out");
    let file = out.0.join("m.oaiybackup");
    let made = make(&data.0, &file);
    let manifest = manifest_of(&file, PASS);
    assert_eq!(manifest.v, 1);
    assert!(chrono::DateTime::parse_from_rfc3339(&manifest.created_at).is_ok());
    assert_eq!(manifest.app.name, "oaiy");
    assert_eq!(manifest.app.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(manifest.platform, std::env::consts::OS);
    assert!(!manifest.includes_keys);
    let names: Vec<&str> = manifest.entries.iter().map(|e| e.name.as_str()).collect();
    for want in ["callers.json", "calendar/calendar.json", "triggers.json", "flows/greeting.json", "setup.json", "voices/receptionist.wav", "plugin-data/aokie/settings.json", "templates/my-rig.json"] {
        assert!(names.contains(&want), "{want} is in the backup");
    }
    for never in ["callers.json.bak", "link/account.json", "ai/providers.json", "ai/codex-home/auth.json", "desktop-e2e-identity.key", "data-node-signing.key", "companion/relay.json", "bridge/pairings.json", "models/model.gguf", "plugins/aokie/manifest.json", "plugin-data/aokie/pairing.json"] {
        assert!(!names.contains(&never), "{never} is not in the backup");
    }
    assert!(names.windows(2).all(|w| w[0] < w[1]), "sorted, so two backups of the same data list alike");
    for entry in &manifest.entries {
        let live = fs::read(data.0.join(&entry.name)).unwrap();
        // The calendar and a plugin's settings are cleaned as they are copied (see sanitize.rs), so what
        // the manifest describes is what went into the backup, not the live bytes.
        if entry.name == "calendar/calendar.json" || entry.name.starts_with("plugin-data/") {
            continue;
        }
        assert_eq!(entry.size as usize, live.len());
        assert_eq!(entry.sha256, sha(&live));
    }
    assert!(manifest.excluded.iter().any(|e| e.pattern == "link/account.json" && e.redo.is_some()));
    assert!(manifest.excluded.iter().any(|e| e.pattern.contains("models/**")));
    assert_eq!(manifest.counts.files as usize, manifest.entries.len());
    assert_eq!(manifest.counts.bytes, manifest.entries.iter().map(|e| e.size).sum::<u64>());
    assert_eq!(manifest.partial.len(), 1, "no Agent page was asked");
    assert_eq!(made.counts, manifest.counts);
    assert!(made.verified);
    assert_eq!(made.excluded, manifest.excluded);
}

#[test]
fn keys_are_added_when_asked_and_the_manifest_says_so() {
    let data = TempDir::new("keys-on");
    realistic(&data.0, "A");
    let out = TempDir::new("keys-on-out");
    let file = out.0.join("k.oaiybackup");
    let made = make_with(&data.0, &file, PASS, true, None).unwrap();
    let manifest = manifest_of(&file, PASS);
    assert!(manifest.includes_keys && made.includes_keys);
    assert!(manifest.entries.iter().any(|e| e.name == "ai/providers.json"));
    assert!(!manifest.entries.iter().any(|e| e.name == "link/account.json"), "the link credential is never included");
    // The file is encrypted like everything else.
    assert!(!String::from_utf8_lossy(&fs::read(&file).unwrap()).contains(PROVIDER_KEY));
}

#[test]
fn a_large_item_streams_through() {
    let data = TempDir::new("large");
    put(&data.0, "callers.json", b"{}");
    // About 6 MiB that does not compress much.
    let mut seed = 0x1234_5678u32;
    let big: Vec<u8> = (0..6 * 1024 * 1024).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 24) as u8
    }).collect();
    put(&data.0, "voices/long.wav", &big);
    let out = TempDir::new("large-out");
    let file = out.0.join("l.oaiybackup");
    make(&data.0, &file);
    let target = TempDir::new("large-target");
    restore::stage(&target.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&target.0), ApplyOutcome::Applied(_)));
    assert_eq!(get(&target.0, "voices/long.wav").unwrap(), big);
}

// ---- the passphrase --------------------------------------------------------------------------------

#[test]
fn a_short_passphrase_is_refused_counting_characters() {
    let data = TempDir::new("short");
    put(&data.0, "callers.json", b"{}");
    let out = TempDir::new("short-out");
    for pass in ["", "short", "elevenchars", "ééééééééééé"] {
        let err = make_with(&data.0, &out.0.join("s.oaiybackup"), pass, false, None).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Passphrase, "{pass:?}");
        assert!(!err.message.contains(pass) || pass.is_empty(), "the message never repeats the passphrase");
    }
    assert!(!out.0.join("s.oaiybackup").exists());
    assert!(make_with(&data.0, &out.0.join("ok.oaiybackup"), "twelve chars", false, None).is_ok());
    assert!(check_passphrase("ééééééééééé").is_err(), "eleven characters are twenty-two bytes");
    assert!(check_passphrase("éééééééééééé").is_ok());
}

#[test]
fn a_wrong_passphrase_is_refused_and_nothing_is_staged() {
    let data = TempDir::new("wrong");
    realistic(&data.0, "A");
    let out = TempDir::new("wrong-out");
    let file = out.0.join("w.oaiybackup");
    make(&data.0, &file);
    let dst = TempDir::new("wrong-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    for op in ["inspect", "stage"] {
        let err = if op == "inspect" {
            restore::inspect(&dst.0, &file, "not the right passphrase", &options()).unwrap_err()
        } else {
            restore::stage(&dst.0, &file, "not the right passphrase", &Ticks::all(), &options()).unwrap_err()
        };
        assert_eq!(err.kind, ErrorKind::WrongPassphrase, "{op}");
        assert!(!err.message.contains("not the right passphrase"));
    }
    assert_eq!(snapshot(&dst.0), before);
    assert_nothing_staged(&dst.0);
}

#[test]
fn a_flipped_byte_anywhere_is_refused_and_nothing_is_staged() {
    let data = TempDir::new("flip");
    realistic(&data.0, "A");
    let out = TempDir::new("flip-out");
    let file = out.0.join("f.oaiybackup");
    make(&data.0, &file);
    let good = fs::read(&file).unwrap();
    let dst = TempDir::new("flip-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    let mut positions: Vec<usize> = vec![0, 1, 5, 20, 21, 22, 40, 60, 100, 150, 180, good.len() - 1, good.len() - 2, good.len() - 16, good.len() - 17];
    positions.extend((1..48).map(|i| i * good.len() / 48));
    positions.sort();
    positions.dedup();
    let bad = out.0.join("bad.oaiybackup");
    for at in positions {
        let mut bytes = good.clone();
        bytes[at] ^= 0x01;
        fs::write(&bad, &bytes).unwrap();
        assert!(restore::inspect(&dst.0, &bad, PASS, &options()).is_err(), "a changed byte at {at} must be refused by the dry run");
        assert!(restore::stage(&dst.0, &bad, PASS, &Ticks::all(), &options()).is_err(), "a changed byte at {at} must be refused by staging");
    }
    assert_eq!(snapshot(&dst.0), before);
    assert_nothing_staged(&dst.0);
}

#[test]
fn a_truncated_file_is_refused_and_nothing_is_staged() {
    let data = TempDir::new("cut");
    realistic(&data.0, "A");
    let out = TempDir::new("cut-out");
    let file = out.0.join("c.oaiybackup");
    make(&data.0, &file);
    let good = fs::read(&file).unwrap();
    let dst = TempDir::new("cut-dst");
    let bad = out.0.join("short.oaiybackup");
    for keep in [0, 1, 10, 30, 100, 150, good.len() / 3, good.len() / 2, good.len() - 65, good.len() - 17, good.len() - 16, good.len() - 1] {
        fs::write(&bad, &good[..keep]).unwrap();
        assert!(restore::inspect(&dst.0, &bad, PASS, &options()).is_err(), "{keep} bytes of {} must be refused (dry run)", good.len());
        assert!(restore::stage(&dst.0, &bad, PASS, &Ticks::all(), &options()).is_err(), "{keep} bytes must be refused (staging)");
    }
    assert!(snapshot(&dst.0).is_empty());
    assert_nothing_staged(&dst.0);
}

#[test]
fn a_file_that_is_not_a_backup_is_refused() {
    let dst = TempDir::new("not-a-backup");
    let junk = dst.0.join("x.oaiybackup");
    fs::write(&junk, b"just some text, not an age file at all").unwrap();
    let err = restore::inspect(&dst.0, &junk, PASS, &options()).unwrap_err();
    assert!(matches!(err.kind, ErrorKind::Damaged | ErrorKind::WrongPassphrase | ErrorKind::Unsupported), "{err}");
    let missing = restore::inspect(&dst.0, &dst.0.join("nope.oaiybackup"), PASS, &options()).unwrap_err();
    assert_eq!(missing.kind, ErrorKind::Io);
}

// ---- hostile backups -------------------------------------------------------------------------------

/// A backup made by hand is refused as a whole, whatever the reason, and leaves nothing behind.
fn assert_refused(dst: &Path, file: &Path, kind: ErrorKind) {
    let before = snapshot(dst);
    let inspect = restore::inspect(dst, file, PASS, &options()).unwrap_err();
    assert_eq!(inspect.kind, kind, "{inspect}");
    let staged = restore::stage(dst, file, PASS, &Ticks::all(), &options()).unwrap_err();
    assert_eq!(staged.kind, kind, "{staged}");
    assert_eq!(snapshot(dst), before);
    assert_nothing_staged(dst);
}

#[test]
fn zip_slip_names_absolute_paths_and_drive_letters_are_refused() {
    for name in ["../evil.json", "voices/../../evil.json", "/etc/evil.json", "\\evil.json", "C:/evil.json", "C:\\evil.json", "voices\\..\\evil.json", "voices//evil.json", "voices/./evil.json", "voices/evil\u{0}.wav"] {
        let out = TempDir::new("slip");
        let files: Vec<(&str, &[u8])> = vec![(name, b"payload")];
        let manifest = manifest_for(&files);
        let file = out.0.join("evil.oaiybackup");
        craft(&file, &manifest, &files, true);
        let dst = TempDir::new("slip-dst");
        put(&dst.0, "callers.json", b"{}");
        assert_refused(&dst.0, &file, ErrorKind::Unsafe);
        assert!(!dst.0.parent().unwrap().join("evil.json").exists(), "nothing was written outside the folder ({name:?})");
    }
}

#[test]
fn duplicate_names_are_refused() {
    let out = TempDir::new("dups");
    // The manifest lists one name twice.
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let mut manifest = manifest_for(&files);
    manifest.entries.push(manifest.entries[0].clone());
    let file = out.0.join("dup.oaiybackup");
    craft(&file, &manifest, &files, true);
    let dst = TempDir::new("dups-dst");
    assert_refused(&dst.0, &file, ErrorKind::Unsafe);
    // Two names that are one file on Windows.
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{}"), ("Callers.json", b"{}")];
    let manifest = manifest_for(&files);
    let file = out.0.join("case.oaiybackup");
    craft(&file, &manifest, &files, true);
    assert_refused(&dst.0, &file, ErrorKind::Unsafe);
}

#[test]
fn a_backup_that_holds_what_a_backup_never_holds_is_refused() {
    for name in ["link/account.json", "desktop-e2e-identity.key", "companion/relay.json", "plugins/aokie/manifest.json", "models/x.gguf", "mystery.bin"] {
        let out = TempDir::new("never");
        let files: Vec<(&str, &[u8])> = vec![(name, b"{\"credential\":\"an attacker's\"}")];
        let manifest = manifest_for(&files);
        let file = out.0.join("never.oaiybackup");
        craft(&file, &manifest, &files, true);
        let dst = TempDir::new("never-dst");
        put(&dst.0, "callers.json", b"{}");
        assert_refused(&dst.0, &file, ErrorKind::Unsafe);
        assert!(!dst.0.join(name).exists());
    }
}

#[test]
fn a_zip_that_does_not_match_its_manifest_is_refused() {
    let out = TempDir::new("mismatch");
    let dst = TempDir::new("mismatch-dst");
    // An entry the manifest does not list.
    let listed: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let manifest = manifest_for(&listed);
    let actual: Vec<(&str, &[u8])> = vec![("callers.json", b"{}"), ("triggers.json", b"{}")];
    let file = out.0.join("extra.oaiybackup");
    craft(&file, &manifest, &actual, true);
    assert_refused(&dst.0, &file, ErrorKind::Unsafe);
    // An entry that is missing.
    let listed: Vec<(&str, &[u8])> = vec![("callers.json", b"{}"), ("triggers.json", b"{}")];
    let manifest = manifest_for(&listed);
    let actual: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let file = out.0.join("missing.oaiybackup");
    craft(&file, &manifest, &actual, true);
    assert_refused(&dst.0, &file, ErrorKind::Damaged);
    // Bytes that are not what the hash says.
    let mut manifest = manifest_for(&[("callers.json", b"{}")]);
    manifest.entries[0].sha256 = sha(b"something else");
    let file = out.0.join("hash.oaiybackup");
    craft(&file, &manifest, &[("callers.json", b"{}")], true);
    assert_refused(&dst.0, &file, ErrorKind::Damaged);
    // A size that is not the size.
    let mut manifest = manifest_for(&[("callers.json", b"{}")]);
    manifest.entries[0].size = 1;
    let file = out.0.join("size.oaiybackup");
    craft(&file, &manifest, &[("callers.json", b"{}")], true);
    assert_refused(&dst.0, &file, ErrorKind::Damaged);
    // The manifest not first.
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let manifest = manifest_for(&files);
    let file = out.0.join("late.oaiybackup");
    craft(&file, &manifest, &files, false);
    assert_refused(&dst.0, &file, ErrorKind::Damaged);
}

#[test]
fn another_version_or_another_program_is_refused() {
    let out = TempDir::new("version");
    let dst = TempDir::new("version-dst");
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let mut manifest = manifest_for(&files);
    manifest.v = 2;
    let file = out.0.join("v2.oaiybackup");
    craft(&file, &manifest, &files, true);
    assert_refused(&dst.0, &file, ErrorKind::Unsupported);
    let mut manifest = manifest_for(&files);
    manifest.app.name = "something-else".into();
    let file = out.0.join("other.oaiybackup");
    craft(&file, &manifest, &files, true);
    assert_refused(&dst.0, &file, ErrorKind::Unsupported);
}

#[test]
fn the_size_and_entry_caps_are_enforced_when_reading() {
    let data = TempDir::new("caps");
    realistic(&data.0, "A");
    let out = TempDir::new("caps-out");
    let file = out.0.join("caps.oaiybackup");
    make(&data.0, &file);
    let dst = TempDir::new("caps-dst");
    let cases: Vec<(&str, Limits)> = vec![
        ("entries", Limits { max_entries: 5, ..Limits::default() }),
        ("one entry", Limits { max_entry_bytes: 100, ..Limits::default() }),
        ("all entries", Limits { max_total_bytes: 500, ..Limits::default() }),
        ("a name", Limits { max_name_len: 8, ..Limits::default() }),
    ];
    for (what, limits) in cases {
        let o = RestoreOptions { limits, ..RestoreOptions::default() };
        let inspect = restore::inspect(&dst.0, &file, PASS, &o).unwrap_err();
        assert!(matches!(inspect.kind, ErrorKind::TooLarge | ErrorKind::Unsafe), "{what}: {inspect}");
        assert!(restore::stage(&dst.0, &file, PASS, &Ticks::all(), &o).is_err(), "{what}");
    }
    assert_nothing_staged(&dst.0);
    // The defaults are the brief's: 200,000 entries.
    assert_eq!(Limits::default().max_entries, 200_000);
}

#[test]
fn the_size_and_entry_caps_are_enforced_when_writing() {
    let data = TempDir::new("caps-w");
    realistic(&data.0, "A");
    let out = TempDir::new("caps-w-out");
    let run = |limits: Limits| {
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let mut o = CreateOptions::new(&data.0, out.0.join("c.oaiybackup"), PASS);
        o.cost = Cost::Fixed(8);
        o.limits = limits;
        create(&o)
    };
    assert_eq!(run(Limits { max_entries: 3, ..Limits::default() }).unwrap_err().kind, ErrorKind::TooLarge);
    assert_eq!(run(Limits { max_total_bytes: 200, ..Limits::default() }).unwrap_err().kind, ErrorKind::TooLarge);
    let made = run(Limits { max_entry_bytes: 100, ..Limits::default() }).unwrap();
    assert!(made.partial.iter().any(|w| w.contains("too large")), "{:?}", made.partial);
    assert!(!manifest_of(&out.0.join("c.oaiybackup"), PASS).entries.iter().any(|e| e.name == "voices/receptionist.wav"));
}

#[test]
fn not_enough_free_space_refuses_before_anything_is_written() {
    let data = TempDir::new("space");
    realistic(&data.0, "A");
    let out = TempDir::new("space-out");
    {
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let mut o = CreateOptions::new(&data.0, out.0.join("s.oaiybackup"), PASS);
        o.cost = Cost::Fixed(8);
        o.free_space = |_| 1024;
        let err = create(&o).unwrap_err();
        assert_eq!(err.kind, ErrorKind::NoSpace);
        assert!(err.message.contains("free space"));
    }
    assert!(fs::read_dir(&out.0).unwrap().next().is_none(), "nothing was made");
    // And a restore checks the disk before it stages.
    let file = out.0.join("ok.oaiybackup");
    make(&data.0, &file);
    let dst = TempDir::new("space-dst");
    let o = RestoreOptions { free_space: |_| 1024, ..RestoreOptions::default() };
    assert_eq!(restore::stage(&dst.0, &file, PASS, &Ticks::all(), &o).unwrap_err().kind, ErrorKind::NoSpace);
    assert_nothing_staged(&dst.0);
}

// ---- refusing while busy: the updater's own decision ---------------------------------------------------------------

/// One thing at a time that is going on, as the updater's activity reports it: what it is, the stable name of the
/// reason the updater gives for it, and a word its sentence has.
fn one_thing_going_on() -> Vec<(&'static str, Box<dyn Fn(&Fake)>, &'static str, &'static str)> {
    fn live() -> LineState {
        LineState::Live { plugin: "Aokie Phone Bridge".into(), count: 1 }
    }
    vec![
        ("a call on OAIY's own line", Box::new(|f: &Fake| f.set(|s| s.calls = 1)), "call", "phone call"),
        ("a call a phone plugin reports (Aokie's own pipeline, which OAIY's line never sees)", Box::new(|f: &Fake| f.set(|s| s.phone = Some(live()))), "phoneCall", "Aokie Phone Bridge"),
        (
            "a phone plugin that cannot say whether a call is live",
            Box::new(|f: &Fake| f.set(|s| s.phone = Some(LineState::Unknown { plugin: "Aokie Phone Bridge".into(), why: "no answer".into() }))),
            "callUnknown",
            "can't tell whether a phone call is live",
        ),
        ("an Agent task from a flow", Box::new(|f: &Fake| f.set(|s| s.tasks = 2)), "agentTask", "2 tasks from flows"),
        ("a download", Box::new(|f: &Fake| f.set(|s| s.downloads = 1)), "download", "downloading"),
        ("a download the engines make", Box::new(|f: &Fake| f.state.lock().unwrap().engines = EnginesState::Known { media: 0, downloads: 2 }), "download", "2 models or files"),
        ("an engine's media job", Box::new(|f: &Fake| f.set(|s| s.media = 1)), "mediaJob", "media"),
        ("engines that are running and cannot say", Box::new(|f: &Fake| f.set(|s| s.engines_unknown = Some("no answer".into()))), "enginesUnknown", "can't tell whether the engines are busy"),
        ("an install", Box::new(|f: &Fake| f.set(|s| s.installing = vec!["Python".into()])), "installing", "Something is installing: Python"),
        ("a move of the data folder", Box::new(|f: &Fake| f.set(|s| s.migrating = true)), "migration", "data folder is being moved"),
    ]
}

#[test]
fn each_thing_that_stops_an_update_stops_a_backup_a_check_and_a_restore_by_itself_in_the_updaters_words() {
    let data = TempDir::new("busy");
    put(&data.0, "callers.json", b"{}");
    let out = TempDir::new("busy-out");
    let file = out.0.join("quiet.oaiybackup");
    make(&data.0, &file);
    for (what, make_busy, code, word) in one_thing_going_on() {
        let activity = Fake::default();
        make_busy(&activity);
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let busy = Busy::look(Some(&activity as &dyn Activity));
        // Every source was asked once, and afresh: this is a decision to go on, not a status a window polls.
        assert_eq!((activity.reads.load(std::sync::atomic::Ordering::SeqCst), activity.fresh_reads.load(std::sync::atomic::Ordering::SeqCst)), (1, 1), "{what}");
        // The same reason, and only it, that the updater gives for it: one decision.
        assert_eq!(busy.codes(), [code], "{what}");
        let updater = blockers::compute(Some(&activity.read(false)), std::time::Duration::from_secs(3600));
        assert_eq!(updater.iter().map(|b| b.code).collect::<Vec<_>>(), [code], "{what}: the updater says the same");
        assert!(busy.reasons()[0].contains(word), "{what}: {:?}", busy.reasons());
        // It refuses a backup ...
        let mut o = CreateOptions::new(&data.0, out.0.join("b.oaiybackup"), PASS);
        o.cost = Cost::Fixed(8);
        o.busy = busy.clone();
        let err = create(&o).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Busy, "{what}");
        assert!(err.message.starts_with("Making a backup has to wait.") && err.message.contains(word) && err.message.ends_with("Try again when it is finished."), "{what}: {err}");
        assert!(!out.0.join("b.oaiybackup").exists(), "{what}: nothing was written");
        // ... the check of a backup, and preparing a restore.
        let restoring = RestoreOptions { busy: busy.clone(), ..RestoreOptions::default() };
        let dst = TempDir::new("busy-dst");
        assert_eq!(restore::inspect(&dst.0, &file, PASS, &restoring).unwrap_err().kind, ErrorKind::Busy, "{what}");
        assert_eq!(restore::stage(&dst.0, &file, PASS, &Ticks::all(), &restoring).unwrap_err().kind, ErrorKind::Busy, "{what}");
        assert_nothing_staged(&dst.0);
    }
}

#[test]
fn a_quiet_app_lets_a_backup_and_a_restore_through_and_an_app_that_has_not_said_yet_does_not() {
    let data = TempDir::new("busy-quiet");
    put(&data.0, "callers.json", b"{}");
    let out = TempDir::new("busy-quiet-out");
    let file = out.0.join("q.oaiybackup");
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // Nothing going on (as the updater's activity reads it, and with no phone plugin): nothing in the way.
    let quiet = Fake::default();
    let busy = Busy::look(Some(&quiet as &dyn Activity));
    assert!(!busy.is_busy() && busy.codes().is_empty() && busy.refuse_if_busy("making a backup").is_ok());
    assert_eq!(busy, Busy::none());
    let mut o = CreateOptions::new(&data.0, &file, PASS);
    o.cost = Cost::Fixed(8);
    o.busy = busy;
    create(&o).expect("a quiet app makes a backup");
    // An idle phone plugin, and an updater that has been up only a moment, do not stand in the way either (the uptime is the update's own).
    let idle = Fake::default();
    idle.set(|s| s.phone = Some(LineState::Idle));
    assert!(!Busy::look(Some(&idle as &dyn Activity)).is_busy());
    // Nothing has said what the app is doing yet: still starting up, and not taken for quiet.
    let starting = Busy::look(None);
    assert_eq!(starting.codes(), ["starting"]);
    assert_eq!(starting.refuse_if_busy("checking a backup").unwrap_err().kind, ErrorKind::Busy);
    // A look that could not be made is not quiet either.
    assert_eq!(Busy::cannot_tell().refuse_if_busy("x").unwrap_err().kind, ErrorKind::Busy);
}

#[test]
fn a_backup_that_is_being_made_is_a_reason_of_the_backups_own_on_top_of_the_updaters() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let activity = Fake::default();
    activity.set(|s| s.tasks = 1);
    let running = state::begin_run().expect("free");
    let busy = Busy::look(Some(&activity as &dyn Activity));
    // The updater's reason first, then the backup's own.
    assert_eq!(busy.codes(), ["agentTask", "backupRunning"]);
    assert!(busy.reasons().join(" ").contains("A backup is being made."));
    drop(running);
    assert_eq!(Busy::look(Some(&activity as &dyn Activity)).codes(), ["agentTask"], "gone when the backup is");
}

#[test]
fn what_a_backup_waits_for_is_worked_out_in_one_place_and_not_a_second_time() {
    // The reasons come from `update::blockers` and nowhere else: no list of sources of a backup's own.
    let sources = [
        ("backup/busy.rs", include_str!("busy.rs")),
        ("backup/create.rs", include_str!("create.rs")),
        ("backup/restore.rs", include_str!("restore.rs")),
        ("backup/commands.rs", include_str!("commands.rs")),
        ("backup/routes.rs", include_str!("routes.rs")),
        ("http.rs", include_str!("../http.rs")),
    ];
    for (file, text) in sources {
        let own_sources: &[&str] = if file == "http.rs" {
            // (The engines' routes and the like are its own business: only the stub that fed a backup is looked for.)
            &["BusySignals", "local_signals", "register_live_calls"]
        } else {
            &["BusySignals", "local_signals", "register_live_calls", "live_call_count", "live_calls()", "pending_count", "engines_ui", "studio_json", "DownloadsHandle", "PythonHandle", "installing_ids", "job_running", "is_installing"]
        };
        for own in own_sources {
            assert!(!text.contains(own), "{file} has a source of its own ({own}): the reasons come from update::blockers");
        }
    }
    let busy = source_text(include_str!("busy.rs"));
    assert!(busy.contains("blockers::work_blockers(readings, WAITS)") && busy.contains("activity.read(true)"), "asked afresh, through the updater's function");
    // The desktop's commands ask the updater's activity, off the runtime's own thread, and take a failed look for busy.
    let look = command_source_of("look_busy");
    assert!(look.contains("try_state::<UpdaterHandle>()") && look.contains("updater.activity()") && look.contains("Busy::look(activity.as_deref())") && look.contains("spawn_blocking") && look.contains("Busy::cannot_tell()"), "{look}");
    // Every command that must not run while the app is busy asks it, and a wait for a dialog does not use an old answer.
    let commands = source_text(include_str!("commands.rs"));
    for (name, refusal) in [("backup_create", "making a backup"), ("backup_restore_inspect", "checking a backup"), ("backup_restart_to_apply", "restarting to finish the restore")] {
        let body = command_source(name);
        assert!(body.contains("look_busy(&app).await") && (body.contains(&format!("refuse_if_busy(\"{refusal}\")"))), "{name} asks whether the app is busy and refuses with {refusal:?}");
    }
    assert!(command_source("backup_restore_stage").contains("busy: look_busy(&app).await"), "preparing a restore takes its own fresh look");
    assert_eq!(commands.matches("gather_busy").count(), 0);
    // The stub that told the backup how many calls the hub had is gone from the window's start-up.
    assert!(!include_str!("../http.rs").contains("backup::busy::register"));
}

/// A source file as text with `\n` line ends, whatever line endings a checkout gave it (a Windows checkout has CRLF).
fn source_text(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// The text of a private function of `commands.rs`, from its `fn` line to the end of its body.
fn command_source_of(name: &str) -> String {
    let source = source_text(include_str!("commands.rs"));
    let start = source.find(&format!("async fn {name}")).unwrap_or_else(|| panic!("{name} is in commands.rs"));
    let rest = &source[start..];
    let end = rest.find("\n}\n").map(|i| i + 3).unwrap_or(rest.len());
    rest[..end].to_string()
}

// ---- restoring: look, stage, apply, undo -----------------------------------------------------------

#[test]
fn a_backup_round_trips_through_the_dry_run_staging_and_the_start_up_apply() {
    let src = TempDir::new("trip-src");
    realistic(&src.0, "A");
    let out = TempDir::new("trip-out");
    let file = out.0.join("trip.oaiybackup");
    make(&src.0, &file);

    let dst = TempDir::new("trip-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);

    // 1. The dry run says what would happen and changes nothing.
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    assert_eq!(snapshot(&dst.0), before, "the dry run changes nothing");
    assert_nothing_staged(&dst.0);
    let of = |id: &str| preview.categories.iter().find(|c| c.id == id).unwrap_or_else(|| panic!("{id} in {:?}", preview.categories)).clone();
    assert_eq!((of("contacts").replaced, of("contacts").added), (1, 0), "the other computer's contacts would be replaced");
    assert_eq!(of("calendar").added, 1);
    assert_eq!((of("flows").added, of("flows").replaced, of("flows").left_alone), (3, 1, 1), "greeting replaced, local-only left alone");
    assert!(preview.lacks.iter().any(|l| l.contains("AI providers")));
    assert!(preview.lacks.iter().any(|l| l.contains("Agent")));
    assert!(preview.redo.iter().any(|r| r.contains("Link FormLogic")), "{:?}", preview.redo);
    assert!(preview.redo.iter().any(|r| r.contains("Pair your phone")));
    assert!(!preview.includes_keys);
    assert_eq!(preview.file_name, "trip.oaiybackup");
    assert!(preview.total_files > 15 && preview.total_bytes > 0);

    // 2. Staging unpacks beside the live data and writes the marker; still nothing live changes.
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert_eq!(staged.kind, "restore");
    assert_eq!(snapshot(&dst.0), before, "staging changes nothing live");
    assert!(dst.0.join("restore").join("pending.json").is_file());
    assert!(dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files").join("callers.json").is_file());
    let info = restore::pending_info(&dst.0).unwrap();
    assert_eq!((info.id.as_str(), info.kind.as_str()), (staged.id.as_str(), "restore"));
    assert_private(&dst.0.join("restore").join("pending.json"));
    assert_private(&dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files").join("callers.json"));
    assert_nothing_left_in_scratch(&dst.0);

    // 3. Applying at the next start.
    let ApplyOutcome::Applied(done) = restore::apply_pending(&dst.0) else { panic!("the restore should be applied") };
    assert!(done.ok && done.kind == "restore");
    assert_eq!(get(&dst.0, "callers.json"), get(&src.0, "callers.json"));
    assert_eq!(json_of(&dst.0, "calendar/calendar.json"), json_of(&src.0, "calendar/calendar.json"));
    assert_eq!(get(&dst.0, "flows/greeting.json"), get(&src.0, "flows/greeting.json"));
    assert_eq!(get(&dst.0, "flows/token-refund.json"), get(&src.0, "flows/token-refund.json"));
    assert_eq!(get(&dst.0, "flows/local-only.json").unwrap(), b"{\"name\":\"only here\"}", "what the backup lacks is left alone");
    assert_eq!(get(&dst.0, "voices/receptionist.wav").unwrap().len(), 4000);
    assert_eq!(json_of(&dst.0, "plugin-data/aokie/settings.json"), json_of(&src.0, "plugin-data/aokie/settings.json"));
    assert_eq!(get(&dst.0, "templates/my-rig.json"), get(&src.0, "templates/my-rig.json"));
    // A restore never touches what a backup leaves out.
    assert_eq!(get(&dst.0, "link/account.json").unwrap(), b"{\"credential\":\"flk_TARGET_OWN\"}");
    assert_eq!(get(&dst.0, "ai/providers.json").unwrap(), b"{\"providers\":[{\"apiKey\":\"sk-TARGET-OWN\"}]}");
    assert_eq!(get(&dst.0, "companion/aokie/endpoint.key").unwrap(), b"TARGET-endpoint");
    assert_eq!(get(&dst.0, "models/keep.gguf").unwrap(), b"keep");
    for never in ["desktop-e2e-identity.key", "data-node-signing.key", "plugins/aokie/manifest.json", "mystery.bin", "bridge/pairings.json", "plugin-data/aokie/pairing.json"] {
        assert!(!dst.0.join(never).exists(), "{never} was not restored");
    }
    assert!(!dst.0.join("restore").join("pending.json").exists(), "the marker is gone");
    assert!(!dst.0.join("restore").join(format!("pending-{}", staged.id)).exists(), "and what was staged");
    assert!(!dst.0.join("restore").join("apply-journal.jsonl").exists());

    // The result is there for the dashboard.
    let last = restore::last_restore(&dst.0).unwrap();
    assert!(last.ok && last.id == staged.id && last.agent_storage == "none");
    assert!(last.redo.iter().any(|r| r.contains("Link FormLogic")));
    let status = state::status(&dst.0);
    assert!(status.pending_restore.is_none() && status.undo_available);

    // Once and only once.
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    let again = snapshot(&dst.0);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    assert_eq!(snapshot(&dst.0), again);
}

fn assert_nothing_left_in_scratch(data: &Path) {
    let scratch = data.join("backup").join("scratch");
    let held: Vec<_> = fs::read_dir(&scratch).map(|d| d.flatten().collect::<Vec<_>>()).unwrap_or_default();
    assert!(held.is_empty(), "no scratch plaintext is left behind");
}

#[test]
fn undo_puts_back_what_was_replaced_and_takes_away_what_was_added() {
    let src = TempDir::new("undo-src");
    realistic(&src.0, "A");
    let out = TempDir::new("undo-out");
    let file = out.0.join("u.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("undo-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);

    assert!(!restore::undo_available(&dst.0));
    assert_eq!(restore::stage_undo(&dst.0, &options()).unwrap_err().kind, ErrorKind::Conflict, "nothing to undo yet");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_ne!(snapshot(&dst.0), before);
    assert!(restore::undo_available(&dst.0));

    // What the restore replaced was moved aside whole, not copied.
    let undo_dir = fs::read_dir(dst.0.join("restore")).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with("undo-")).unwrap().path();
    assert_eq!(fs::read(undo_dir.join("files").join("callers.json")).unwrap(), b"{\"contacts\":[{\"name\":\"Somebody else\"}]}");

    let staged = restore::stage_undo(&dst.0, &options()).unwrap();
    assert_eq!(staged.kind, "undo");
    assert!(state::status(&dst.0).pending_restore.is_some_and(|p| p.kind == "undo"));
    // Applied at the next start, the same way.
    let ApplyOutcome::Applied(done) = restore::apply_pending(&dst.0) else { panic!("the undo should be applied") };
    assert_eq!(done.kind, "undo");
    assert_eq!(snapshot(&dst.0), before, "everything is as it was before the restore");
    // An undo keeps a snapshot of what it replaced and took away, so it can be put back (a redo).
    assert!(restore::undo_available(&dst.0), "the undo's own snapshot is kept");
    assert_eq!(restore::undo_kind(&dst.0).as_deref(), Some("undo"));
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
}

/// Restore a backup onto a folder that has files of its own, then stage the undo: returns the folder
/// as it was before, as it is after the restore, and the folder itself.
fn restored_and_undo_staged() -> (TempDir, BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>) {
    let src = TempDir::new("undo-fail-src");
    realistic(&src.0, "A");
    let out = TempDir::new("undo-fail-out");
    let file = out.0.join("u.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("undo-fail-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let restored = snapshot(&dst.0);
    restore::stage_undo(&dst.0, &options()).unwrap();
    (dst, before, restored)
}

#[test]
fn an_undo_that_fails_or_crashes_never_loses_the_saved_copies() {
    // Two files are put back and the rest taken away: fail after the second is set aside, and in the taking away.
    for (what, inject) in [
        ("fails after a file was put back", Inject::FailBeforeInstall(1)),
        ("fails in the first file", Inject::FailBeforeInstall(0)),
        ("fails while taking away what the restore added", Inject::FailBeforeInstall(2)),
        ("crashes after a file was put back", Inject::CrashBeforeInstall(1)),
        ("crashes while taking away what the restore added", Inject::CrashBeforeInstall(3)),
    ] {
        let (dst, before, restored) = restored_and_undo_staged();
        restore::INJECT.with(|c| c.set(Some(inject)));
        let outcome = restore::apply_pending(&dst.0);
        restore::INJECT.with(|c| c.set(None));
        if matches!(inject, Inject::CrashBeforeInstall(_)) {
            assert!(matches!(outcome, ApplyOutcome::None), "{what}");
            assert_ne!(snapshot(&dst.0), restored, "{what}: the crash left it half done");
            assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Failed(_)), "{what}: the next start rolls it back");
        } else {
            assert!(matches!(outcome, ApplyOutcome::Failed(_)), "{what}");
        }
        assert_eq!(snapshot(&dst.0), restored, "{what}: the folder is exactly as the restore left it");
        assert!(restore::undo_available(&dst.0), "{what}: the saved copies are still there");
        assert!(dst.0.join("restore").read_dir().unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("undo-") && e.path().join("files").join("callers.json").is_file()), "{what}: the original callers.json is still in its snapshot");
        // And the undo still works, whole, the next time.
        restore::stage_undo(&dst.0, &options()).unwrap();
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)), "{what}");
        assert_eq!(snapshot(&dst.0), before, "{what}: the second undo brings back everything");
    }
}

#[test]
fn only_the_last_two_undo_snapshots_are_kept() {
    let src = TempDir::new("keep-src");
    realistic(&src.0, "A");
    let out = TempDir::new("keep-out");
    let file = out.0.join("k.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("keep-dst");
    let mut ids = Vec::new();
    for _ in 0..4 {
        let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
        ids.push(staged.id);
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut kept: Vec<String> = fs::read_dir(dst.0.join("restore")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("undo-")).collect();
    kept.sort();
    let mut want = vec![format!("undo-{}", ids[2]), format!("undo-{}", ids[3])];
    want.sort();
    assert_eq!(kept, want, "the two newest stay");
}

#[test]
fn cancelling_a_staged_restore_leaves_nothing() {
    let src = TempDir::new("cancel-src");
    realistic(&src.0, "A");
    let out = TempDir::new("cancel-out");
    let file = out.0.join("c.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("cancel-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    restore::discard_pending(&dst.0).unwrap();
    assert!(restore::pending_info(&dst.0).is_none());
    assert_nothing_staged(&dst.0);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    assert_eq!(snapshot(&dst.0), before);
    restore::discard_pending(&dst.0).unwrap();
}

#[test]
fn staging_again_replaces_the_earlier_staged_restore() {
    let src = TempDir::new("again-src");
    realistic(&src.0, "A");
    let out = TempDir::new("again-out");
    let file = out.0.join("a.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("again-dst");
    let first = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let second = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert_ne!(first.id, second.id);
    assert!(!dst.0.join("restore").join(format!("pending-{}", first.id)).exists());
    assert_eq!(restore::pending_info(&dst.0).unwrap().id, second.id);
}

#[test]
fn a_failure_midway_puts_everything_back_and_is_reported() {
    let src = TempDir::new("fail-src");
    realistic(&src.0, "A");
    let out = TempDir::new("fail-out");
    let file = out.0.join("f.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("fail-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    // The third file fails after the one it replaces was set aside.
    restore::INJECT.with(|c| c.set(Some(Inject::FailBeforeInstall(3))));
    let outcome = restore::apply_pending(&dst.0);
    restore::INJECT.with(|c| c.set(None));
    let ApplyOutcome::Failed(last) = outcome else { panic!("the restore should fail") };
    assert!(!last.ok && last.error.as_deref().is_some_and(|e| e.contains("put back")), "{last:?}");
    assert_eq!(snapshot(&dst.0), before, "every file is as it was");
    assert!(!dst.0.join("restore").join("pending.json").exists());
    assert!(!dst.0.join("restore").join("apply-journal.jsonl").exists());
    assert!(!restore::undo_available(&dst.0), "there is nothing to undo");
    // It is reported on the next start, and not repeated.
    let reported = restore::last_restore(&dst.0).unwrap();
    assert!(!reported.ok && reported.error.is_some());
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    // The staged copies do not stay behind: they are personal data.
    assert!(fs::read_dir(dst.0.join("restore")).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().starts_with("pending-")));
}

#[test]
fn a_crash_during_the_apply_is_rolled_back_at_the_next_start() {
    let src = TempDir::new("crash-src");
    realistic(&src.0, "A");
    let out = TempDir::new("crash-out");
    let file = out.0.join("c.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("crash-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    // The process "dies" after set-aside of the fourth file, before its replacement is put in.
    restore::INJECT.with(|c| c.set(Some(Inject::CrashBeforeInstall(3))));
    let outcome = restore::apply_pending(&dst.0);
    restore::INJECT.with(|c| c.set(None));
    assert!(matches!(outcome, ApplyOutcome::None));
    assert_ne!(snapshot(&dst.0), before, "the crash left things half done");
    assert!(dst.0.join("restore").join("apply-journal.jsonl").is_file());
    // The next start finds the journal and puts everything back.
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("the interrupted restore should be rolled back") };
    assert!(last.error.as_deref().unwrap().contains("interrupted"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
}

#[test]
fn a_crash_after_the_last_file_is_finished_at_the_next_start() {
    let src = TempDir::new("done-src");
    realistic(&src.0, "A");
    let out = TempDir::new("done-out");
    let file = out.0.join("d.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("done-dst");
    target(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    restore::INJECT.with(|c| c.set(Some(Inject::CrashAfterDone)));
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    restore::INJECT.with(|c| c.set(None));
    assert!(dst.0.join("restore").join("pending.json").is_file(), "the marker is removed last");
    let ApplyOutcome::Applied(done) = restore::apply_pending(&dst.0) else { panic!("finished") };
    assert!(done.ok);
    assert_eq!(get(&dst.0, "callers.json"), get(&src.0, "callers.json"));
    assert!(restore::undo_available(&dst.0), "and the undo record is kept");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
}

#[test]
fn a_changed_staged_file_or_marker_is_not_applied() {
    let src = TempDir::new("tamper-src");
    realistic(&src.0, "A");
    let out = TempDir::new("tamper-out");
    let file = out.0.join("t.oaiybackup");
    make(&src.0, &file);

    // A staged file changed after staging: to another length, and to the same length.
    let dst = TempDir::new("tamper-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let files = dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files");
    put(&files, "callers.json", b"{\"contacts\":[\"planted\"]}");
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(last.error.as_deref().unwrap().contains("staged"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before);
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let files = dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files");
    let mut same_length = fs::read(files.join("callers.json")).unwrap();
    let last_byte = same_length.len() - 3;
    same_length[last_byte] ^= 0x01;
    fs::write(files.join("callers.json"), same_length).unwrap();
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(last.error.as_deref().unwrap().contains("does not check out"), "a changed byte at the same length is caught by its hash: {last:?}");
    assert_eq!(snapshot(&dst.0), before);

    // A marker that names a place outside the folder.
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let marker_path = dst.0.join("restore").join("pending.json");
    let mut marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(&marker_path).unwrap()).unwrap();
    marker["files"][0]["name"] = serde_json::json!("../evil.json");
    fs::write(&marker_path, marker.to_string()).unwrap();
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(last.error.as_deref().unwrap().contains("refused"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before);
    assert!(!dst.0.parent().unwrap().join("evil.json").exists());
    let _ = staged;

    // A marker that names a credential, with a staged file there that matches its own hash: only the
    // marker's own check stands between it and the live folder.
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let files = dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files");
    let planted = b"{\"credential\":\"an attacker's\"}";
    put(&files, "link/account.json", planted);
    let mut marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(&marker_path).unwrap()).unwrap();
    marker["files"].as_array_mut().unwrap().push(serde_json::json!({ "name": "link/account.json", "size": planted.len(), "sha256": sha(planted) }));
    fs::write(&marker_path, marker.to_string()).unwrap();
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(last.error.as_deref().unwrap().contains("refused"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before, "the credential that was there is untouched");
    assert_eq!(get(&dst.0, "link/account.json").unwrap(), b"{\"credential\":\"flk_TARGET_OWN\"}");

    // A marker that asks for the removal of a credential (an undo takes away what a restore added).
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let mut marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(&marker_path).unwrap()).unwrap();
    marker["removals"] = serde_json::json!(["link/account.json", "callers.json"]);
    fs::write(&marker_path, marker.to_string()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Failed(_)));
    assert_eq!(snapshot(&dst.0), before, "nothing was taken away");

    // A marker that is not a marker.
    fs::write(&marker_path, b"{ not json").unwrap();
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(last.error.as_deref().unwrap().contains("damaged"));
    assert_eq!(snapshot(&dst.0), before);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
}

#[test]
fn a_link_in_the_way_stops_staging_and_stops_the_apply() {
    let outside = TempDir::new("way-outside");
    let src = TempDir::new("way-src");
    put(&src.0, "callers.json", b"{}");
    put(&src.0, "plugin-data/aokie/settings.json", b"{\"x\":1}");
    let out = TempDir::new("way-out");
    let file = out.0.join("w.oaiybackup");
    make(&src.0, &file);

    // A junction where a folder of the restore would go: refused at staging.
    let dst = TempDir::new("way-dst");
    fs::create_dir_all(dst.0.join("plugin-data")).unwrap();
    let Some(link) = crate::plugins::trust::tests::dir_link(&dst.0.join("plugin-data").join("aokie"), &outside.0) else {
        eprintln!("no directory link could be made here: nothing to test");
        return;
    };
    let err = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsafe, "{err}");
    assert!(fs::read_dir(&outside.0).unwrap().next().is_none(), "nothing was written through the link");
    assert_nothing_staged(&dst.0);
    drop(link);

    // A junction that appears after staging: refused at the apply, and everything is put back.
    let dst = TempDir::new("way-dst2");
    put(&dst.0, "callers.json", b"{\"mine\":true}");
    fs::create_dir_all(dst.0.join("plugin-data")).unwrap();
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let _link = crate::plugins::trust::tests::dir_link(&dst.0.join("plugin-data").join("aokie"), &outside.0).unwrap();
    let ApplyOutcome::Failed(last) = restore::apply_pending(&dst.0) else { panic!("refused") };
    assert!(!last.ok, "{last:?}");
    assert!(fs::read_dir(&outside.0).unwrap().next().is_none(), "nothing was written through the link");
    assert_eq!(get(&dst.0, "callers.json").unwrap(), b"{\"mine\":true}", "and what was already put in place was put back");
    let _ = before;
}

// ---- when the self-check fails ---------------------------------------------------------------------

#[test]
fn a_backup_that_does_not_check_out_after_writing_is_deleted_and_reported() {
    let data = TempDir::new("verify");
    realistic(&data.0, "A");
    let out = TempDir::new("verify-out");
    let final_name = out.0.join("v.oaiybackup");
    // An earlier backup of that name must survive a failed one.
    fs::write(&final_name, b"the earlier backup").unwrap();
    create::CORRUPT_OUTPUT.with(|c| c.set(true));
    let err = make_with(&data.0, &final_name, PASS, false, None).unwrap_err();
    create::CORRUPT_OUTPUT.with(|c| c.set(false));
    assert_eq!(err.kind, ErrorKind::Verify);
    assert!(err.message.contains("deleted"), "{err}");
    assert_eq!(fs::read(&final_name).unwrap(), b"the earlier backup", "the earlier backup is untouched");
    let names: Vec<String> = fs::read_dir(&out.0).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["v.oaiybackup"], "no half-written file is left");
    assert_nothing_left_in_scratch(&data.0);
    let status = state::status(&data.0);
    assert_eq!(status.last_backup_ok, Some(false));
    // The next one works and replaces it.
    make_with(&data.0, &final_name, PASS, false, None).unwrap();
    assert!(fs::read(&final_name).unwrap().starts_with(b"age-encryption.org/v1"));
    assert_eq!(state::status(&data.0).last_backup_ok, Some(true));
}

#[test]
fn the_status_says_when_and_how_big_the_last_backup_was() {
    let data = TempDir::new("status");
    put(&data.0, "callers.json", b"{}");
    let empty = state::status(&data.0);
    assert!(empty.last_backup_at.is_none() && empty.last_backup_ok.is_none() && empty.pending_restore.is_none() && !empty.undo_available);
    let out = TempDir::new("status-out");
    let made = make(&data.0, &out.0.join("s.oaiybackup"));
    let status = state::status(&data.0);
    assert_eq!(status.last_backup_at.as_deref(), Some(made.created_at.as_str()));
    assert_eq!(status.last_backup_ok, Some(true));
    assert_eq!(status.last_backup_size, Some(made.size));
    let json = serde_json::to_value(&status).unwrap();
    for key in ["lastBackupAt", "lastBackupOk", "lastBackupSize", "pendingRestore", "lastRestore", "undoAvailable", "running"] {
        assert!(json.get(key).is_some(), "{key}");
    }
    let text = json.to_string();
    assert!(!text.contains("s.oaiybackup") && !text.contains(&out.0.display().to_string()), "no path in the status");
}

// ---- the Agent's storage ---------------------------------------------------------------------------

fn agent_zip() -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    writer.start_file("agent-manifest.json", opts).unwrap();
    writer.write_all(b"{\"v\":1,\"kind\":\"oaiy-agent-storage\"}").unwrap();
    writer.start_file("opfs/projects/p1/chat.json", opts).unwrap();
    writer.write_all(b"[{\"role\":\"user\",\"text\":\"hello\"}]").unwrap();
    writer.finish().unwrap().into_inner()
}

/// A page that answers: posts its ZIP in parts, then says it is done.
struct Page {
    zip: Vec<u8>,
    part_size: usize,
    ok: bool,
    warnings: Vec<String>,
}

impl AgentExport for Page {
    fn request(&self, id: &str, token: &str, _include_keys: bool) -> std::result::Result<(), String> {
        let (id, token, zip, part_size, ok, warnings) = (id.to_string(), token.to_string(), self.zip.clone(), self.part_size, self.ok, self.warnings.clone());
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let mut parts = 0u32;
            if ok {
                for chunk in zip.chunks(part_size) {
                    agent::receive_part(&id, &token, parts, chunk).unwrap();
                    parts += 1;
                }
            }
            let done = DonePayload {
                ok,
                error: (!ok).then(|| "it went wrong".to_string()),
                parts,
                counts: agent::ExportCounts { projects: 1, incognito_skipped: 1, conversations: 1, files: 2, bytes: 100 },
                warnings,
            };
            agent::finish(&id, &token, done).unwrap();
        });
        Ok(())
    }
}

struct NoPage;
impl AgentExport for NoPage {
    fn request(&self, _: &str, _: &str, _: bool) -> std::result::Result<(), String> {
        Err("the Agent's page is not open".into())
    }
}

struct SilentPage;
impl AgentExport for SilentPage {
    fn request(&self, _: &str, _: &str, _: bool) -> std::result::Result<(), String> {
        Ok(())
    }
}

#[test]
fn the_agents_storage_is_included_when_its_page_answers() {
    let data = TempDir::new("agent");
    realistic(&data.0, "A");
    let out = TempDir::new("agent-out");
    let file = out.0.join("a.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: 40, ok: true, warnings: vec!["one file was too large".into()] };
    let made = make_with(&data.0, &file, PASS, false, Some(&page)).unwrap();
    assert_eq!((made.counts.agent_projects, made.counts.agent_conversations, made.counts.agent_files), (1, 1, 2));
    assert!(!made.partial.iter().any(|w| w == MISSING_WARNING), "{:?}", made.partial);
    assert!(made.partial.iter().any(|w| w.contains("one file was too large")), "the page's own warnings are passed on");
    let manifest = manifest_of(&file, PASS);
    let entry = manifest.entries.iter().find(|e| e.name == "agent/agent-storage.zip").expect("the Agent's storage is an entry");
    assert_eq!(entry.sha256, sha(&page.zip));
    // The parts arrived whole and in order: the backup holds exactly what the page sent.
    let zip = plain_zip(&file, PASS);
    let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
    let mut inner = Vec::new();
    archive.by_name("agent/agent-storage.zip").unwrap().read_to_end(&mut inner).unwrap();
    assert_eq!(inner, page.zip);
    assert_nothing_left_in_scratch(&data.0);
}

#[test]
fn a_missing_silent_or_failing_agent_page_makes_the_backup_partial_but_it_completes() {
    let data = TempDir::new("agent-missing");
    realistic(&data.0, "A");
    let out = TempDir::new("agent-missing-out");
    let failing = Page { zip: agent_zip(), part_size: 40, ok: false, warnings: vec![] };
    let cases: Vec<(&str, &dyn AgentExport)> = vec![("not open", &NoPage), ("silent", &SilentPage), ("failing", &failing)];
    for (what, page) in cases {
        let file = out.0.join(format!("{what}.oaiybackup").replace(' ', "-"));
        let made = make_with(&data.0, &file, PASS, false, Some(page)).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(made.partial.iter().any(|w| w == MISSING_WARNING), "{what}: {:?}", made.partial);
        assert_eq!(MISSING_WARNING, "Agent conversations and projects were not included: open the Agent and try again");
        let manifest = manifest_of(&file, PASS);
        assert!(!manifest.entries.iter().any(|e| e.name.starts_with("agent/")), "{what}");
        assert!(manifest.partial.iter().any(|w| w == MISSING_WARNING), "{what}: the manifest says so too");
        assert!(manifest.entries.iter().any(|e| e.name == "callers.json"), "{what}: the rest of the backup is whole");
    }
}

#[test]
fn an_export_session_takes_parts_only_with_its_token_in_order_and_within_limits() {
    let dir = TempDir::new("session");
    let path = dir.0.join("agent.part");
    let (id, token) = agent::open_session(&path, 100).unwrap();
    assert_eq!(agent::receive_part(&id, "not-the-token", 0, b"x"), Err(PartError::Denied));
    assert_eq!(agent::receive_part("no-such-session", &token, 0, b"x"), Err(PartError::Unknown));
    assert_eq!(agent::receive_part(&id, &token, 1, b"x"), Err(PartError::Sequence), "out of order");
    assert_eq!(agent::receive_part(&id, &token, 0, b"abc"), Ok(()));
    assert_eq!(agent::receive_part(&id, &token, 0, b"abc"), Err(PartError::Sequence), "a part twice");
    assert_eq!(agent::receive_part(&id, &token, 1, &vec![0u8; 98]), Err(PartError::TooLarge), "past the session's limit");
    assert_eq!(agent::receive_part(&id, &token, 1, &vec![0u8; PART_SIZE + 1]), Err(PartError::TooLarge), "a part over 4 MiB");
    let done = DonePayload { ok: true, parts: 1, ..Default::default() };
    assert_eq!(agent::finish(&id, "wrong", done.clone()), Err(PartError::Denied));
    assert_eq!(agent::finish(&id, &token, done.clone()), Ok(()));
    assert_eq!(agent::receive_part(&id, &token, 1, b"more"), Err(PartError::Closed), "nothing after done");
    assert_eq!(fs::read(&path).unwrap(), b"abc");
    agent::close_session(&id);
    assert_eq!(agent::receive_part(&id, &token, 1, b"more"), Err(PartError::Unknown));
    assert_ne!(token, agent::open_session(&dir.0.join("other.part"), 10).unwrap().1, "every session has its own token");
}

#[test]
fn a_restore_hands_the_agents_storage_to_its_page_and_takes_the_undo_snapshot_back() {
    let src = TempDir::new("import-src");
    realistic(&src.0, "A");
    let out = TempDir::new("import-out");
    let file = out.0.join("i.oaiybackup");
    // Big enough for three 4 MiB parts.
    let mut big = agent_zip();
    big.extend((0..(9 * 1024 * 1024)).map(|i| (i % 251) as u8));
    let page = Page { zip: big.clone(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();

    let dst = TempDir::new("import-dst");
    target(&dst.0);
    assert!(!agent::import_meta(&dst.0).pending);
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(staged.agent_storage);
    assert!(!agent::import_meta(&dst.0).pending, "not until it is applied at the next start");
    let ApplyOutcome::Applied(done) = restore::apply_pending(&dst.0) else { panic!("applied") };
    assert_eq!(done.agent_storage, "pending");
    assert_eq!(restore::last_restore(&dst.0).unwrap().agent_storage, "pending");

    // The page asks, and is told what waits, how to fetch it and its token.
    let meta = agent::import_meta(&dst.0);
    assert!(meta.pending);
    let (id, token, size) = (meta.id.clone().unwrap(), agent::page_token().to_string(), meta.size.unwrap());
    assert_eq!(id, staged.id);
    assert_eq!(meta.kind.as_deref(), Some("restore"));
    assert_eq!(meta.part_size, Some(PART_SIZE as u64));
    assert_eq!(meta.parts, Some(size.div_ceil(PART_SIZE as u64)));
    assert_eq!(meta.parts, Some(3));
    assert_eq!(size as usize, big.len());
    assert_eq!(meta.sha256.as_deref(), Some(sha(&big).as_str()));
    assert_eq!(agent::import_part(&dst.0, &id, "not-the-token", 0), Err(PartError::Denied));
    assert_eq!(agent::import_part(&dst.0, "0000000000000000", &token, 0), Err(PartError::Unknown));
    assert_eq!(agent::import_part(&dst.0, &id, &token, 3), Err(PartError::Sequence));
    let mut got = Vec::new();
    for i in 0..3 {
        let part = agent::import_part(&dst.0, &id, &token, i).unwrap();
        assert!(part.len() <= PART_SIZE);
        got.extend(part);
    }
    assert_eq!(got, big, "the parts add up to the storage");

    // The page saves what it holds now, for the undo, before it imports.
    assert_eq!(agent::undo_part(&dst.0, &id, &token, 1, b"out of order"), Err(PartError::Sequence));
    assert_eq!(agent::undo_part(&dst.0, &id, "wrong", 0, b"x"), Err(PartError::Denied));
    agent::undo_part(&dst.0, &id, &token, 0, b"snapshot-part-0;").unwrap();
    agent::undo_part(&dst.0, &id, &token, 1, b"snapshot-part-1").unwrap();
    agent::undo_done(&dst.0, &id, &token, &DonePayload { ok: true, parts: 2, ..Default::default() }).unwrap();
    let snapshot_path = agent::undo_agent_path(&dst.0, &id);
    assert_eq!(fs::read(&snapshot_path).unwrap(), b"snapshot-part-0;snapshot-part-1");
    assert_private(&snapshot_path);

    // It says how it went, and the storage is not offered again.
    agent::import_done(&dst.0, &id, &token, &agent::ImportReport { ok: true, ..Default::default() }).unwrap();
    assert!(!agent::import_meta(&dst.0).pending);
    assert_eq!(restore::last_restore(&dst.0).unwrap().agent_storage, "applied");

    // An undo hands the snapshot back to the page the same way (with no snapshot of its own).
    let undo = restore::stage_undo(&dst.0, &options()).unwrap();
    assert!(undo.agent_storage);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let meta = agent::import_meta(&dst.0);
    assert_eq!(meta.kind.as_deref(), Some("undo"));
    assert_eq!(agent::import_part(&dst.0, meta.id.as_deref().unwrap(), &token, 0).unwrap(), b"snapshot-part-0;snapshot-part-1");
    agent::import_done(&dst.0, meta.id.as_deref().unwrap(), &token, &agent::ImportReport { ok: false, error: Some("no room".into()), ..Default::default() }).unwrap();
    let last = restore::last_restore(&dst.0).unwrap();
    assert_eq!(last.agent_storage, "failed");
    assert!(last.redo.iter().any(|r| r.contains("no room")));
}

// ---- the routes ------------------------------------------------------------------------------------

/// A request as the Agent's own page makes it.
async fn call(app: &axum::Router, method: &str, path: &str, token: Option<&str>, body: Vec<u8>) -> (u16, Vec<u8>) {
    call_from(app, method, path, token, Some("http://oaiy.localhost"), body).await
}

async fn call_from(app: &axum::Router, method: &str, path: &str, token: Option<&str>, origin: Option<&str>, body: Vec<u8>) -> (u16, Vec<u8>) {
    use tower::ServiceExt as _;
    let mut request = axum::http::Request::builder().method(method).uri(path);
    if let Some(o) = origin {
        request = request.header("origin", o);
    }
    if let Some(t) = token {
        request = request.header("x-backup-token", t);
    }
    if method == "POST" && !body.is_empty() && path.ends_with("done") {
        request = request.header("content-type", "application/json");
    }
    let response = app.clone().oneshot(request.body(axum::body::Body::from(body)).unwrap()).await.unwrap();
    let status = response.status().as_u16();
    (status, axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec())
}

#[tokio::test]
async fn no_http_route_can_create_or_restore_a_backup() {
    let dir = TempDir::new("routes");
    let app = routes::router(dir.0.clone());
    // The routes there are: one read-only status route and the Agent page's hand-over routes.
    assert_eq!(routes::ROUTES.len(), 8);
    assert_eq!(routes::ROUTES.iter().filter(|(m, _)| *m == "GET").map(|(_, p)| *p).collect::<Vec<_>>(), ["/api/backup/status", "/api/backup/agent-import", "/api/backup/agent-import/:id/part/:index"]);
    for (method, path) in [
        ("POST", "/api/backup"), ("GET", "/api/backup"), ("POST", "/api/backup/create"), ("POST", "/api/backup/run"), ("POST", "/api/backup/restore"), ("PUT", "/api/backup/restore"),
        ("POST", "/api/backup/stage"), ("POST", "/api/backup/apply"), ("POST", "/api/backup/undo"), ("DELETE", "/api/backup/restore"), ("POST", "/api/backup/verify"),
        ("POST", "/api/backup/status"), ("PUT", "/api/backup/status"), ("DELETE", "/api/backup/status"),
        ("POST", "/api/backup/agent-import"), ("POST", "/api/backup/agent"), ("GET", "/api/backup/agent/x/part"),
    ] {
        let (status, _) = call(&app, method, path, Some("x"), b"{}".to_vec()).await;
        assert!(status == 404 || status == 405, "{method} {path} answered {status}");
    }
    assert!(!dir.0.join("backup").join("status.json").exists() && !dir.0.join("restore").exists(), "and none of it changed anything");
}

#[tokio::test]
async fn the_status_route_reports_and_the_hand_over_routes_need_their_tokens() {
    let dir = TempDir::new("routes2");
    put(&dir.0, "callers.json", b"{}");
    let app = routes::router(dir.0.clone());
    let (status, body) = call(&app, "GET", "/api/backup/status", None, Vec::new()).await;
    assert_eq!(status, 200);
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["lastBackupAt"].is_null() && json["pendingRestore"].is_null() && json["undoAvailable"] == false);

    // No import waits (the page asks with the secret the desktop gave it; without it, nothing is answered).
    assert_eq!(call(&app, "GET", "/api/backup/agent-import", None, Vec::new()).await.0, 403);
    let (status, body) = call(&app, "GET", "/api/backup/agent-import", Some(agent::page_token()), Vec::new()).await;
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap(), serde_json::json!({ "pending": false }));

    // A session the desktop opened.
    let (id, token) = agent::open_session(&dir.0.join("route.part"), 8 << 20).unwrap();
    let part = format!("/api/backup/agent/{id}/part?seq=0");
    assert_eq!(call(&app, "POST", &part, None, b"abc".to_vec()).await.0, 403, "no token");
    assert_eq!(call(&app, "POST", &part, Some("wrong"), b"abc".to_vec()).await.0, 403);
    assert_eq!(call(&app, "POST", "/api/backup/agent/nope/part?seq=0", Some(&token), b"abc".to_vec()).await.0, 404);
    assert_eq!(call(&app, "POST", &format!("/api/backup/agent/{id}/part?seq=5"), Some(&token), b"abc".to_vec()).await.0, 409);
    assert_eq!(call(&app, "POST", &part, Some(&token), vec![0u8; PART_SIZE + 100 * 1024]).await.0, 413, "the body limit");
    assert_eq!(call(&app, "POST", &part, Some(&token), vec![1u8; PART_SIZE]).await.0, 200, "a part of exactly 4 MiB");
    let done = serde_json::json!({ "ok": true, "parts": 1, "counts": { "projects": 1 }, "warnings": [] }).to_string().into_bytes();
    assert_eq!(call(&app, "POST", &format!("/api/backup/agent/{id}/done"), Some("wrong"), done.clone()).await.0, 403);
    assert_eq!(call(&app, "POST", &format!("/api/backup/agent/{id}/done"), Some(&token), done).await.0, 200);
    assert_eq!(fs::metadata(dir.0.join("route.part")).unwrap().len(), PART_SIZE as u64);
    agent::close_session(&id);
}

#[tokio::test]
async fn the_hand_over_routes_are_for_the_agents_own_page_only() {
    let dir = TempDir::new("routes-origin");
    let app = routes::router(dir.0.clone());
    let (id, token) = agent::open_session(&dir.0.join("o.part"), 1 << 20).unwrap();
    let part = format!("/api/backup/agent/{id}/part?seq=0");
    let done = serde_json::json!({ "ok": true, "parts": 1 }).to_string().into_bytes();
    for origin in [None, Some("https://oaiy.com"), Some("https://app.oaiy.com"), Some("tauri://localhost"), Some("http://tauri.localhost"), Some("http://oaiyflows.localhost"), Some("http://oaiy.localhost.evil.example"), Some("http://127.0.0.1:17973"), Some("null")] {
        // Even with the right session token, and whatever the desktop's guard would have let through.
        assert_eq!(call_from(&app, "POST", &part, Some(&token), origin, b"abc".to_vec()).await.0, 403, "part from {origin:?}");
        assert_eq!(call_from(&app, "POST", &format!("/api/backup/agent/{id}/done"), Some(&token), origin, done.clone()).await.0, 403, "done from {origin:?}");
        assert_eq!(call_from(&app, "GET", "/api/backup/agent-import", Some(&token), origin, Vec::new()).await.0, 403, "import from {origin:?}");
        assert_eq!(call_from(&app, "GET", "/api/backup/agent-import/x/part/0", Some(&token), origin, Vec::new()).await.0, 403, "import part from {origin:?}");
        for tail in ["undo-part?seq=0", "undo-part?seq=1"] {
            assert_eq!(call_from(&app, "POST", &format!("/api/backup/agent-import/x/{tail}"), Some(&token), origin, b"abc".to_vec()).await.0, 403, "{tail} from {origin:?}");
        }
        for tail in ["undo-done", "done"] {
            assert_eq!(call_from(&app, "POST", &format!("/api/backup/agent-import/x/{tail}"), Some(&token), origin, done.clone()).await.0, 403, "{tail} from {origin:?}");
        }
    }
    assert_eq!(fs::metadata(dir.0.join("o.part")).unwrap().len(), 0, "nothing was stored for any of them");
    // The Agent's page, in each of its forms, is let in.
    for origin in ["http://oaiy.localhost", "https://oaiy.localhost", "oaiy://localhost"] {
        assert!(routes::is_agent_origin(origin));
        assert_eq!(call_from(&app, "GET", "/api/backup/agent-import", Some(agent::page_token()), Some(origin), Vec::new()).await.0, 200, "{origin}");
    }
    assert!(!routes::is_agent_origin("http://oaiy.localhost/") && !routes::is_agent_origin(""));
    // The status is not the Agent's alone: the dashboard reads it.
    assert_eq!(call_from(&app, "GET", "/api/backup/status", None, Some("tauri://localhost"), Vec::new()).await.0, 200);
    assert_eq!(call_from(&app, "GET", "/api/backup/status", None, None, Vec::new()).await.0, 200);
    agent::close_session(&id);
}

// ---- nothing secret leaks --------------------------------------------------------------------------

static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        CAPTURED.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{} {}", record.target(), record.args()));
    }
    fn flush(&self) {}
}

fn capture_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

#[test]
fn no_passphrase_or_secret_appears_in_a_log_line_or_an_error() {
    capture_logs();
    let src = TempDir::new("leak-src");
    realistic(&src.0, "A");
    let out = TempDir::new("leak-out");
    let mut messages: Vec<String> = Vec::new();
    fn note(messages: &mut Vec<String>, what: &str, r: std::result::Result<(), BackupError>) {
        match r {
            Err(e) => messages.push(format!("{what}: {e} / {e:?}")),
            Ok(()) => panic!("{what} should have failed"),
        }
    }

    // Making backups, with and without keys, that work and that fail.
    let file = out.0.join("leak.oaiybackup");
    let made = make_with(&src.0, &file, CANARY_PASS, true, None);
    messages.push(format!("{:?}", made.as_ref().map(|m| (&m.path, &m.partial, &m.excluded, &m.counts))));
    assert!(made.is_ok());
    note(&mut messages, "short", make_with(&src.0, &out.0.join("x.oaiybackup"), "CANARYshort", false, None).map(|_| ()));
    create::CORRUPT_OUTPUT.with(|c| c.set(true));
    note(&mut messages, "corrupt", make_with(&src.0, &out.0.join("y.oaiybackup"), CANARY_PASS, true, None).map(|_| ()));
    create::CORRUPT_OUTPUT.with(|c| c.set(false));

    // Reading them, rightly and wrongly.
    let dst = TempDir::new("leak-dst");
    target(&dst.0);
    let preview = restore::inspect(&dst.0, &file, CANARY_PASS, &options());
    messages.push(format!("{:?}", preview.as_ref().map(|p| (&p.categories, &p.redo, &p.partial))));
    note(&mut messages, "wrong inspect", restore::inspect(&dst.0, &file, "the wrong passphrase CANARY", &options()).map(|_| ()));
    note(&mut messages, "wrong stage", restore::stage(&dst.0, &file, "the wrong passphrase CANARY", &Ticks::all(), &options()).map(|_| ()));
    let bad = out.0.join("bad.oaiybackup");
    let mut bytes = fs::read(&file).unwrap();
    let n = bytes.len();
    bytes[n / 2] ^= 1;
    fs::write(&bad, bytes).unwrap();
    note(&mut messages, "damaged", restore::inspect(&dst.0, &bad, CANARY_PASS, &options()).map(|_| ()));
    let staged = restore::stage(&dst.0, &file, CANARY_PASS, &Ticks::all(), &options());
    messages.push(format!("{staged:?}"));
    assert!(staged.is_ok());
    restore::INJECT.with(|c| c.set(Some(Inject::FailBeforeInstall(1))));
    let outcome = restore::apply_pending(&dst.0);
    restore::INJECT.with(|c| c.set(None));
    messages.push(format!("{outcome:?}"));
    restore::stage(&dst.0, &file, CANARY_PASS, &Ticks::all(), &options()).unwrap();
    messages.push(format!("{:?}", restore::apply_pending(&dst.0)));
    messages.push(serde_json::to_string(&state::status(&dst.0)).unwrap());

    let logged = CAPTURED.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
    // The scan below means nothing if nothing was captured: this run must have logged its own lines.
    assert!(logged.contains("is staged and will be applied at the next start"), "the logger did not capture this run: {logged:?}");
    assert!(logged.contains("did not finish and was rolled back"), "{logged:?}");
    let everything = format!("{}\n{logged}", messages.join("\n"));
    for canary in [CANARY_PASS, "CANARYshort", LINK_KEY, PROVIDER_KEY, "flk_TARGET_OWN", "sk-TARGET-OWN", "the wrong passphrase CANARY", "CCCCendpoint", "AAAAtunnel"] {
        assert!(!everything.contains(canary), "{canary:?} must not appear in a log line or an error");
    }

    // The passphrase and the secrets are not in any file the backup left behind either, except where a
    // secret is meant to be (the restored provider keys, in their own private file).
    for data in [&src.0, &dst.0] {
        for (name, body) in snapshot_all(data) {
            let text = String::from_utf8_lossy(&body);
            assert!(!text.contains(CANARY_PASS), "the passphrase is in {name}");
        }
    }
    assert_nothing_left_in_scratch(&src.0);
    assert_nothing_left_in_scratch(&dst.0);
}

/// Every file under `root`, including the backup's and restore's own folders.
fn snapshot_all(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            let meta = fs::symlink_metadata(&path).unwrap();
            if meta.is_dir() {
                walk(root, &path, out);
            } else if meta.is_file() {
                out.insert(path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

#[test]
fn every_file_the_backup_writes_is_private_from_its_first_byte() {
    let src = TempDir::new("private-src");
    realistic(&src.0, "A");
    let out = TempDir::new("private-out");
    let file = out.0.join("p.oaiybackup");
    make_with(&src.0, &file, PASS, true, None).unwrap();
    assert_private(&file);
    assert_private(&src.0.join("backup").join("status.json"));
    let dst = TempDir::new("private-dst");
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks { keys: true, ..Ticks::all() }, &options()).unwrap();
    let root = dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files");
    for (name, _) in snapshot_all(&root) {
        assert_private(&root.join(name));
    }
    assert_private(&dst.0.join("restore").join("pending.json"));
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_private(&dst.0.join("ai").join("providers.json"));
    let undo = fs::read_dir(dst.0.join("restore")).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with("undo-")).unwrap().path();
    assert_private(&undo.join("undo.json"));
}

// ---- the dashboard's commands ----------------------------------------------------------------------

#[cfg(feature = "gui")]
#[test]
fn only_the_dashboards_own_window_may_call_the_commands() {
    use super::commands::{check_label, export_script};
    assert!(check_label("main").is_ok());
    for label in ["embed-agent", "embed-flows", "embed-engines", "", "Main", "main2", "mainx"] {
        assert!(check_label(label).is_err(), "{label:?}");
    }
    let script = export_script("abc123", "def456", true);
    assert_eq!(script, "window.__oaiyBackup && window.__oaiyBackup.export({ id: \"abc123\", token: \"def456\", includeKeys: true });");
    assert!(export_script("a", "b", false).contains("includeKeys: false"));
}

#[test]
fn what_a_killed_backup_or_restore_leaves_behind_is_swept_at_the_start() {
    let src = TempDir::new("sweep-src");
    realistic(&src.0, "A");
    let out = TempDir::new("sweep-out");
    let file = out.0.join("s.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("sweep-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    // What a killed run leaves: a working copy of the whole data set, a staging that never got its marker,
    // and the set-aside folder of an undo.
    put(&dst.0.join("backup").join("scratch").join("aaaaaaaaaaaaaaaa"), "tree/f000001", b"a plaintext copy");
    put(&dst.0.join("backup").join("scratch").join("aaaaaaaaaaaaaaaa"), "plain.zip", b"a decrypted backup");
    put(&dst.0.join("restore").join("pending-bbbbbbbbbbbbbbbb"), "files/callers.json", b"orphan");
    fs::create_dir_all(dst.0.join("restore").join("undo-cccccccccccccccc").join("files").join("sub")).unwrap();
    put(&dst.0.join("restore").join("undo-eeeeeeeeeeeeeeee"), "files/callers.json", b"the only copy of something");
    put(&dst.0.join("restore").join("undo-dddddddddddddddd"), "files/callers.json", b"a snapshot");
    put(&dst.0.join("restore").join("agent-import"), "current.zip", b"waits for the page");
    let restore_dir = dst.0.join("restore");
    let pending = restore_dir.join(format!("pending-{}", staged.id));

    // With a marker waiting: its staging stays, so does everything that is not a leftover, and an undo's
    // set-aside folder stays too (a rollback of an interrupted apply needs it).
    let removed = restore::sweep_leftovers(&dst.0);
    assert_eq!(removed, 2, "the working copy and the unnamed staging");
    assert!(!dst.0.join("backup").join("scratch").join("aaaaaaaaaaaaaaaa").exists());
    assert!(!restore_dir.join("pending-bbbbbbbbbbbbbbbb").exists());
    assert!(pending.join("files").join("callers.json").is_file(), "what the marker names stays");
    assert!(restore_dir.join("undo-cccccccccccccccc").exists(), "while a marker waits a set-aside folder may be needed");
    assert!(restore_dir.join("undo-dddddddddddddddd").exists() && restore_dir.join("agent-import").join("current.zip").is_file());
    assert_eq!(restore::sweep_leftovers(&dst.0), 0, "and there is nothing more to sweep");

    // The staged restore still applies.
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(get(&dst.0, "callers.json"), get(&src.0, "callers.json"));

    // With no marker an empty set-aside folder goes too; one that holds a file, a snapshot and the Agent's import do not.
    assert_eq!(restore::sweep_leftovers(&dst.0), 1);
    assert!(!restore_dir.join("undo-cccccccccccccccc").exists());
    assert_eq!(fs::read(restore_dir.join("undo-eeeeeeeeeeeeeeee").join("files").join("callers.json")).unwrap(), b"the only copy of something");
    assert!(restore_dir.join("undo-dddddddddddddddd").exists() && restore_dir.join("agent-import").join("current.zip").is_file());
    let _ = before;
}

#[test]
fn a_backup_run_is_taken_one_at_a_time() {
    let data = TempDir::new("one");
    put(&data.0, "callers.json", b"{}");
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let guard = state::begin_run().expect("free");
    let out = TempDir::new("one-out");
    // Made directly, not through the test lock: a run is already taking the place.
    let mut o = CreateOptions::new(&data.0, out.0.join("o.oaiybackup"), PASS);
    o.cost = Cost::Fixed(8);
    let err = create(&o).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Conflict);
    drop(guard);
    assert!(state::running().is_none());
}

// ---- a hostile file must not be able to keep the check busy ----------------------------------------

/// Write a file that starts like an age file and then does what `body` says.
fn hostile_header(path: &Path, body: &[u8]) {
    let mut bytes = b"age-encryption.org/v1\n".to_vec();
    bytes.extend_from_slice(body);
    fs::write(path, bytes).unwrap();
}

fn assert_refused_quickly(dst: &Path, file: &Path, what: &str) {
    let started = std::time::Instant::now();
    let err = restore::inspect(dst, file, PASS, &options()).unwrap_err();
    assert!(matches!(err.kind, ErrorKind::Damaged | ErrorKind::Unsupported), "{what}: {err}");
    assert!(started.elapsed() < std::time::Duration::from_secs(3), "{what} took {:?}: age must never be handed a header this long", started.elapsed());
    let started = std::time::Instant::now();
    assert!(restore::stage(dst, file, PASS, &Ticks::all(), &options()).is_err(), "{what}");
    assert!(started.elapsed() < std::time::Duration::from_secs(3), "{what} (staging) took {:?}", started.elapsed());
    assert_nothing_staged(dst);
}

#[test]
fn an_oversized_or_unterminated_header_is_refused_before_age_reads_it() {
    let dst = TempDir::new("headers");
    let file = dst.0.join("h.oaiybackup");
    // One line that never ends, a megabyte long.
    let mut body = b"-> scrypt ".to_vec();
    body.extend(std::iter::repeat(b'A').take(1 << 20));
    hostile_header(&file, &body);
    assert_refused_quickly(&dst.0, &file, "a megabyte with no newline");
    // A megabyte of short stanza lines that does end: what makes age's parser quadratic.
    let mut body = Vec::new();
    while body.len() < (1 << 20) {
        body.extend_from_slice(b"-> x\nQUJD\n");
    }
    body.extend_from_slice(b"--- QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU\n");
    hostile_header(&file, &body);
    assert_refused_quickly(&dst.0, &file, "a megabyte of stanzas that ends properly");
    // A terminator just past the cap.
    let mut body = b"-> scrypt AAAAAAAAAAAAAAAAAAAAAA 8\n".to_vec();
    body.extend(std::iter::repeat(b'A').take(container::MAX_HEADER_BYTES));
    body.extend_from_slice(b"\n--- QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU\n");
    hostile_header(&file, &body);
    assert_refused_quickly(&dst.0, &file, "a terminator beyond the cap");
    // Nothing but the version line, and not an age file at all.
    hostile_header(&file, b"");
    assert_refused_quickly(&dst.0, &file, "only the version line");
    fs::write(&file, vec![b'x'; 1 << 20]).unwrap();
    assert_refused_quickly(&dst.0, &file, "not an age file");
    // A real header is far under the cap.
    let src = TempDir::new("headers-src");
    put(&src.0, "callers.json", b"{}");
    let real = dst.0.join("real.oaiybackup");
    make(&src.0, &real);
    let bytes = fs::read(&real).unwrap();
    let end = bytes.windows(5).position(|w| w == b"\n--- ").unwrap();
    assert!(end < 300, "a passphrase header is {end} bytes");
    assert!(end * 4 < container::MAX_HEADER_BYTES);
    assert!(restore::inspect(&dst.0, &real, PASS, &options()).is_ok());
}

#[test]
fn a_check_that_runs_out_of_time_is_stopped_and_leaves_nothing() {
    let src = TempDir::new("time-src");
    realistic(&src.0, "A");
    let out = TempDir::new("time-out");
    let file = out.0.join("t.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("time-dst");
    let slow = RestoreOptions { time_limit: std::time::Duration::ZERO, ..RestoreOptions::default() };
    assert_eq!(restore::inspect(&dst.0, &file, PASS, &slow).unwrap_err().kind, ErrorKind::Timeout);
    assert_eq!(restore::stage(&dst.0, &file, PASS, &Ticks::all(), &slow).unwrap_err().kind, ErrorKind::Timeout);
    assert_nothing_staged(&dst.0);
    // With time it works, and the defaults are minutes, not hours.
    assert!(restore::inspect(&dst.0, &file, PASS, &options()).is_ok());
    assert!(RestoreOptions::default().time_limit <= std::time::Duration::from_secs(30 * 60));
}

#[test]
fn looking_at_a_backup_or_staging_one_waits_while_the_app_is_busy() {
    let src = TempDir::new("busy-restore-src");
    put(&src.0, "callers.json", b"{}");
    let out = TempDir::new("busy-restore-out");
    let file = out.0.join("b.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("busy-restore-dst");
    let busy = RestoreOptions { busy: Busy::from_readings(&Readings { hub_calls: 1, ..Readings::default() }), ..RestoreOptions::default() };
    assert_eq!(restore::inspect(&dst.0, &file, PASS, &busy).unwrap_err().kind, ErrorKind::Busy);
    assert_eq!(restore::stage(&dst.0, &file, PASS, &Ticks::all(), &busy).unwrap_err().kind, ErrorKind::Busy);
    assert_nothing_staged(&dst.0);
}

#[test]
fn scrypt_work_above_the_cap_is_refused_before_any_of_it_is_done() {
    let src = TempDir::new("cap-src");
    put(&src.0, "callers.json", b"{}");
    let out = TempDir::new("cap-out");
    let file = out.0.join("c.oaiybackup");
    make(&src.0, &file);
    let good = fs::read(&file).unwrap();
    // The header says "-> scrypt <salt> 8": make it ask for more than 2^20.
    let line_end = good.iter().position(|b| *b == b'\n').unwrap() + 1;
    let stanza_end = good[line_end..].iter().position(|b| *b == b'\n').unwrap() + line_end;
    assert_eq!(&good[stanza_end - 2..stanza_end], b" 8", "the test file was written at work factor 8");
    assert_eq!(container::MAX_READ_WORK_FACTOR, 20);
    let dst = TempDir::new("cap-dst");
    for asked in [21u32, 22, 23, 30, 60, 99] {
        let mut bytes = good[..stanza_end - 1].to_vec();
        bytes.extend_from_slice(asked.to_string().as_bytes());
        bytes.extend_from_slice(&good[stanza_end..]);
        let bad = out.0.join("bad.oaiybackup");
        fs::write(&bad, bytes).unwrap();
        let started = std::time::Instant::now();
        let err = restore::inspect(&dst.0, &bad, PASS, &options()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported, "work factor {asked}: {err}");
        assert!(started.elapsed() < std::time::Duration::from_secs(2), "work factor {asked} was refused only after {:?}", started.elapsed());
    }
    assert_nothing_staged(&dst.0);
}

#[test]
fn the_work_factor_a_backup_is_written_with_has_a_floor_and_a_ceiling() {
    use container::{default_work_factor, pick_work_factor, MAX_WRITE_WORK_FACTOR as MAX, MIN_WRITE_WORK_FACTOR as MIN};
    assert_eq!((MIN, MAX), (18, 20));
    assert!(MAX <= container::MAX_READ_WORK_FACTOR, "a file OAIY writes is one OAIY opens");
    // A slow or busy computer times a weak factor: it is lifted to the floor.
    for seconds in [10.0, 1.0, 0.5, 0.2, 0.1, 0.06] {
        assert_eq!(pick_work_factor(seconds), MIN, "{seconds}");
    }
    // A fast one is let go higher, never past the ceiling.
    assert_eq!(pick_work_factor(0.03), 19);
    assert_eq!(pick_work_factor(0.015), 20);
    for seconds in [0.005, 0.001, 0.0, -1.0] {
        assert_eq!(pick_work_factor(seconds), MAX, "{seconds}");
    }
    let picked = default_work_factor();
    assert!((MIN..=MAX).contains(&picked), "{picked}");
}

/// Run by hand (it takes a second and up to a gigabyte): a backup at the default cost writes a work
/// factor in range, and opens.
#[test]
#[ignore]
fn a_backup_at_the_default_cost_is_written_within_the_allowed_work_factors() {
    let src = TempDir::new("default-cost-src");
    put(&src.0, "callers.json", b"{}");
    let out = TempDir::new("default-cost-out");
    let file = out.0.join("d.oaiybackup");
    {
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let o = CreateOptions::new(&src.0, &file, PASS);
        create(&o).unwrap();
    }
    let bytes = fs::read(&file).unwrap();
    let text = String::from_utf8_lossy(&bytes[..200]).to_string();
    let factor: u8 = text.lines().nth(1).unwrap().rsplit(' ').next().unwrap().parse().unwrap();
    assert!((18..=20).contains(&factor), "{factor}");
    assert!(restore::inspect(&out.0, &file, PASS, &options()).is_ok());
}

// ---- plugin data is opt-in and never carries a PIN or a key ---------------------------------------

const AOKIE_SETTINGS: &str = r#"{
  "greeting": "hello",
  "businessHours": { "open": "09:00", "close": "17:00" },
  "managerPin": "dpapi1:QUFBQQ==",
  "managerNumber": "0491 570 156",
  "nested": { "pinCode": "1234", "ok": true, "sealedNote": "dpapi:zzzz" },
  "list": ["keep", "dpapi1:qqqq"],
  "apiToken": "sk-not-a-real-token-0003",
  "mapping": "kept",
  "keyword": "kept too"
}"#;

#[test]
fn a_sensitive_key_is_recognised_by_its_words() {
    for key in ["managerPin", "manager_pin", "apiKey", "API_KEY", "sessionToken", "pairingCode", "privateKey", "clientSecret", "PIN", "pin", "authToken", "dpapiBlob", "Password", "manager-auth"] {
        assert!(sanitize::is_sensitive_key(key), "{key} names something secret");
    }
    for key in ["mapping", "typing", "keyword", "keywords", "greeting", "businessHours", "spinner", "pinned", "opening", "author"] {
        assert!(!sanitize::is_sensitive_key(key), "{key} is an ordinary setting");
    }
}

#[test]
fn the_managers_pin_and_other_sealed_values_never_travel_in_a_backup() {
    let data = TempDir::new("pin");
    put(&data.0, "callers.json", b"{}");
    put(&data.0, "plugin-data/aokie/settings.json", AOKIE_SETTINGS);
    put(&data.0, "plugin-data/aokie/settings.json.bak", AOKIE_SETTINGS);
    put(&data.0, "plugin-data/aokie/manager-auth.json", b"{\"failedAttempts\":3,\"lockedUntil\":123}");
    put(&data.0, "plugin-data/aokie/consent.json", b"{\"consented\":true}");
    put(&data.0, "plugin-data/aokie/aokie_radio/pairing_store.json", b"{\"phone\":\"AA:BB:CC\"}");
    put(&data.0, "plugin-data/aokie/outbox.db", b"sealed rows");
    put(&data.0, "plugin-data/another/settings.json", b"{\"a\":1}");
    let out = TempDir::new("pin-out");
    let file = out.0.join("p.oaiybackup");
    let made = make_with(&data.0, &file, PASS, true, None).unwrap();
    let manifest = manifest_of(&file, PASS);
    let plugin_entries: Vec<&str> = manifest.entries.iter().map(|e| e.name.as_str()).filter(|n| n.starts_with("plugin-data/")).collect();
    assert_eq!(plugin_entries, ["plugin-data/aokie/settings.json"], "only the listed file of the listed plugin");
    // What is in it is the settings without the PIN, the token and everything sealed.
    let zip = plain_zip(&file, PASS);
    let mut archive = zip::ZipArchive::new(Cursor::new(zip.clone())).unwrap();
    let mut text = String::new();
    archive.by_name("plugin-data/aokie/settings.json").unwrap().read_to_string(&mut text).unwrap();
    let kept: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(kept["greeting"], "hello");
    assert_eq!(kept["businessHours"]["open"], "09:00");
    assert_eq!(kept["nested"]["ok"], true);
    assert_eq!(kept["list"], serde_json::json!(["keep"]));
    assert_eq!(kept["mapping"], "kept");
    assert_eq!(kept["keyword"], "kept too");
    for gone in ["managerPin", "managerNumber", "apiToken"] {
        assert!(kept.get(gone).is_none(), "{gone} is not in the backup");
    }
    assert!(kept["nested"].get("pinCode").is_none() && kept["nested"].get("sealedNote").is_none());
    // No entry holds any of it. (The manifest, entry 0, names what was left out by key, never by value.)
    for canary in ["dpapi", "1234", "sk-not-a-real-token-0003", "QUFBQQ", "AA:BB:CC", "failedAttempts"] {
        for i in 1..archive.len() {
            let mut body = String::new();
            let _ = archive.by_index(i).unwrap().read_to_string(&mut body);
            assert!(!body.contains(canary), "{canary} must not be in the backup (entry {i})");
        }
    }
    let manifest_text = serde_json::to_string(&manifest).unwrap();
    for canary in ["sk-not-a-real-token-0003", "QUFBQQ", "AA:BB:CC", "1234"] {
        assert!(!manifest_text.contains(canary), "{canary} must not be in the manifest either");
    }
    // Each thing left out is listed with a reason.
    let listed = |pattern: &str| made.excluded.iter().any(|e| e.pattern == pattern);
    for pattern in ["plugin-data/aokie/settings.json: managerPin", "plugin-data/aokie/settings.json: managerNumber", "plugin-data/aokie/settings.json: nested.pinCode", "plugin-data/aokie/settings.json: apiToken", "plugin-data/aokie/**", "plugin-data/another/"] {
        assert!(listed(pattern), "{pattern} is listed: {:?}", made.excluded.iter().map(|e| e.pattern.as_str()).collect::<Vec<_>>());
    }
    assert!(made.excluded.iter().find(|e| e.pattern.ends_with("managerPin")).unwrap().redo.is_some());
}

#[test]
fn a_hostile_settings_file_cannot_plant_a_pin_and_this_computers_own_is_kept() {
    let out = TempDir::new("plant");
    let planted = br#"{"greeting":"attacker's","managerPin":"1111","token":"attacker","nested":{"pinCode":"9999","ok":true}}"#;
    let files: Vec<(&str, &[u8])> = vec![("plugin-data/aokie/settings.json", planted)];
    let file = out.0.join("plant.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    // A computer that has its own sealed PIN keeps it; the rest of the settings come from the backup.
    let dst = TempDir::new("plant-dst");
    put(&dst.0, "plugin-data/aokie/settings.json", br#"{"greeting":"mine","managerPin":"dpapi1:LOCALSEALED","nested":{"pinCode":"local-pin"}}"#);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let got = json_of(&dst.0, "plugin-data/aokie/settings.json");
    assert_eq!(got["greeting"], "attacker's");
    assert_eq!(got["managerPin"], "dpapi1:LOCALSEALED", "this computer's own PIN is not replaced");
    assert_eq!(got["nested"]["pinCode"], "local-pin");
    assert_eq!(got["nested"]["ok"], true);
    assert!(got.get("token").is_none());
    // A computer with none gets none.
    let bare = TempDir::new("plant-bare");
    restore::stage(&bare.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&bare.0), ApplyOutcome::Applied(_)));
    let got = json_of(&bare.0, "plugin-data/aokie/settings.json");
    assert!(got.get("managerPin").is_none() && got.get("token").is_none() && got["nested"].get("pinCode").is_none());
    // A settings file that is not JSON is not brought back at all.
    let files: Vec<(&str, &[u8])> = vec![("plugin-data/aokie/settings.json", b"not json at all"), ("callers.json", b"{}")];
    let junk = out.0.join("junk.oaiybackup");
    craft(&junk, &manifest_for(&files), &files, true);
    let dst = TempDir::new("plant-junk");
    put(&dst.0, "plugin-data/aokie/settings.json", b"{\"mine\":true}");
    restore::stage(&dst.0, &junk, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(json_of(&dst.0, "plugin-data/aokie/settings.json"), serde_json::json!({ "mine": true }), "the file that could not be cleaned was left out");
    assert!(dst.0.join("callers.json").exists());
}

#[test]
fn a_plugin_that_is_not_on_the_list_and_a_file_that_is_not_listed_are_refused() {
    let out = TempDir::new("policy");
    for name in ["plugin-data/other/settings.json", "plugin-data/aokie/manager-auth.json", "plugin-data/aokie/aokie_radio/pairing_store.json", "plugin-data/aokie/notes.txt", "plugin-data/aokie/settings.json.bak"] {
        let files: Vec<(&str, &[u8])> = vec![(name, b"{}")];
        let file = out.0.join("p.oaiybackup");
        craft(&file, &manifest_for(&files), &files, true);
        let dst = TempDir::new("policy-dst");
        assert_refused(&dst.0, &file, ErrorKind::Unsafe);
        assert!(!dst.0.join(name).exists(), "{name}");
    }
}

#[test]
fn ntfs_short_names_cannot_get_a_file_past_the_word_and_extension_rules() {
    let limits = Limits::default();
    for name in [
        "plugin-data/aokie/aokie_radio/PAIRIN~1.JSO", "plugin-data/aokie/AUTH~1.JSO", "plugin-data/aokie/LINK~1.DPA", "plugin-data/aokie/pairin~1.jso", "flows/A~2", "~1", "a/b~9/c.json",
        "plugin-data/aokie/SETTIN~1.JSO",
    ] {
        assert!(container::check_entry_name(name, &limits).is_err(), "{name} is a short-name alias");
    }
    for name in ["a~b.txt", "flows/tilde~.json", "flows/a~.json"] {
        assert!(container::check_entry_name(name, &limits).is_ok(), "{name} is not one");
    }
    // A backup (or a marker) that names one is refused as a whole.
    let out = TempDir::new("short");
    for name in ["plugin-data/aokie/aokie_radio/PAIRIN~1.JSO", "plugin-data/aokie/AUTH~1.JSO", "plugin-data/aokie/LINK~1.DPA", "flows/GREETI~1.JSO"] {
        let files: Vec<(&str, &[u8])> = vec![(name, b"{\"paired\":\"ATTACKER\"}")];
        let file = out.0.join("s.oaiybackup");
        craft(&file, &manifest_for(&files), &files, true);
        let dst = TempDir::new("short-dst");
        assert_refused(&dst.0, &file, ErrorKind::Unsafe);
    }
    // And a place that answers to a short name is not written through: the name a file has in its
    // folder is compared with the name asked for (where the volume has short names at all).
    let dir = TempDir::new("short-target");
    put(&dir.0, "plugin-data/aokie/aokie_radio/pairing_store.json", b"{\"paired\":\"REAL\"}");
    let alias = dir.0.join("plugin-data/aokie/aokie_radio/PAIRIN~1.JSO");
    if fs::symlink_metadata(&alias).is_ok() {
        let err = restore::check_target(&dir.0, "plugin-data/aokie/aokie_radio/PAIRIN~1.JSO").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsafe, "{err}");
        assert!(err.message.contains("short-name alias"));
    } else {
        eprintln!("this volume has no short names: only the name check could be tested");
    }
    // The real name is fine, in any case.
    assert!(restore::check_target(&dir.0, "plugin-data/aokie/aokie_radio/pairing_store.json").is_ok());
    assert!(restore::check_target(&dir.0, "plugin-data/aokie/aokie_radio/PAIRING_STORE.JSON").is_ok(), "the same name in another case is the same file");
}

// ---- the calendar's FormLogic sync state stays where it was ----------------------------------------

const CALENDAR_WITH_SYNC: &str = r#"{
  "settings": { "hours": "9-5" },
  "appointments": [
    { "id": "a1", "title": "Check-up", "notes": "bring the form", "formlogic": { "id": "remote-1", "etag": "e1", "syncedAt": "2026-01-01" } },
    { "id": "a2", "title": "Follow-up" }
  ],
  "deleted": [ { "id": "d1", "formlogicId": "remote-9", "requestKey": "oaiy:d1", "deletedAt": "2026-01-02", "checked": false } ],
  "sync": { "form": "form-123", "cursor": "2026-01-03 00:00:00", "lastSuccessAt": "2026-01-03", "discard": ["r1", "r2"] }
}"#;

#[test]
fn the_calendars_formlogic_sync_state_is_not_backed_up_and_not_restored() {
    let data = TempDir::new("calendar");
    put(&data.0, "calendar/calendar.json", CALENDAR_WITH_SYNC);
    let out = TempDir::new("calendar-out");
    let file = out.0.join("c.oaiybackup");
    make(&data.0, &file);
    let zip = plain_zip(&file, PASS);
    let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
    let mut text = String::new();
    archive.by_name("calendar/calendar.json").unwrap().read_to_string(&mut text).unwrap();
    let kept: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(kept["settings"]["hours"], "9-5");
    assert_eq!(kept["appointments"].as_array().unwrap().len(), 2);
    assert_eq!(kept["appointments"][0]["title"], "Check-up");
    assert_eq!(kept["appointments"][0]["notes"], "bring the form");
    for gone in ["sync", "deleted"] {
        assert!(kept.get(gone).is_none(), "{gone} is not in the backup");
    }
    assert!(kept["appointments"][0].get("formlogic").is_none());
    for canary in ["form-123", "remote-1", "remote-9", "oaiy:d1", "2026-01-03"] {
        assert!(!text.contains(canary), "{canary}");
    }
    // A hostile backup that carries the state has it removed on the way in.
    let files: Vec<(&str, &[u8])> = vec![("calendar/calendar.json", CALENDAR_WITH_SYNC.as_bytes())];
    let hostile = out.0.join("hostile.oaiybackup");
    craft(&hostile, &manifest_for(&files), &files, true);
    let dst = TempDir::new("calendar-dst");
    restore::stage(&dst.0, &hostile, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let got = json_of(&dst.0, "calendar/calendar.json");
    assert_eq!(got["appointments"].as_array().unwrap().len(), 2);
    assert!(got.get("sync").is_none() && got.get("deleted").is_none() && got["appointments"][0].get("formlogic").is_none());
}

// ---- what can run or reconfigure things comes back only when it was ticked ------------------------

const EVIL_TEMPLATE: &str = r#"{"id":"evil","name":"Totally Legit","description":"x","category":"x","defaultPort":9999,"autostart":true,"run":{"command":"cmd.exe","args":["/c","calc.exe"]}}"#;
const LEGIT_TEMPLATE: &str = r#"{"id":"my-rig","name":"My rig","description":"mine","category":"LLM","defaultPort":8123,"run":{"command":"python","args":["server.py","--port","${port}"]}}"#;

/// The services the registry would load from `data`, and which of them start with the app.
fn registry_view(data: &Path) -> Vec<(String, bool)> {
    let registry = crate::services::registry::Registry::init(data.to_path_buf(), data.join("models"), Vec::new()).expect("a registry");
    let mut out: Vec<(String, bool)> = registry.snapshot().services.into_iter().map(|s| (s.id, s.autostart)).collect();
    out.sort();
    out
}

fn ticks_of(classes: &[RestoreClass], keys: bool) -> Ticks {
    Ticks { classes: classes.iter().copied().collect(), keys }
}

#[test]
fn the_reviewers_evil_template_and_autostart_are_refused_by_default_and_shown_by_name() {
    let out = TempDir::new("evil");
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{\"contacts\":[]}"), ("services-autostart.json", b"[\"evil\"]"), ("templates/evil.json", EVIL_TEMPLATE.as_bytes())];
    let file = out.0.join("evil.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    let dst = TempDir::new("evil-dst");
    let before = registry_view(&dst.0);
    assert!(!before.iter().any(|(id, _)| id == "evil"));

    // The dry run names it and says what it does.
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let template = preview.items.iter().find(|i| i.name == "templates/evil.json").expect("the template is listed by name");
    assert_eq!(template.class, RestoreClass::Templates);
    assert!(template.what.contains("cmd.exe") && template.what.contains("calc.exe"), "it shows the program: {}", template.what);
    assert!(template.what.contains("STARTS with OAIY"), "{}", template.what);
    let autostart = preview.items.iter().find(|i| i.name == "services-autostart.json").expect("the autostart entry is listed");
    assert_eq!(autostart.title, "evil");
    assert!(autostart.what.contains("Starts with OAIY at every start") && autostart.what.contains("in this backup"), "{}", autostart.what);
    let class = preview.classes.iter().find(|c| c.id == "templates").expect("the class is offered");
    assert_eq!(class.count, 2);
    assert!(class.description.contains("program"), "{}", class.description);

    // By default (nothing ticked) it does not come back, and the registry does not load it.
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::none(), &options()).unwrap();
    assert!(staged.skipped.iter().any(|l| l.contains("Service templates") && l.contains("not ticked")), "{:?}", staged.skipped);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert!(dst.0.join("callers.json").exists(), "the data came back");
    assert!(!dst.0.join("templates").join("evil.json").exists(), "the template did not");
    assert!(!dst.0.join("services-autostart.json").exists(), "and neither did the autostart list");
    assert!(!registry_view(&dst.0).iter().any(|(id, _)| id == "evil"));

    // Ticking something else (flows, settings, providers) does not bring it back either.
    let other = ticks_of(&[RestoreClass::Flows, RestoreClass::Settings, RestoreClass::Providers, RestoreClass::Plugins, RestoreClass::Connections], true);
    restore::stage(&dst.0, &file, PASS, &other, &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert!(!dst.0.join("templates").join("evil.json").exists());
    assert!(!registry_view(&dst.0).iter().any(|(id, _)| id == "evil"));

    // Only the explicit tick brings it back (and then the person has seen its program listed).
    restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Templates], false), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert!(registry_view(&dst.0).contains(&("evil".to_string(), true)), "ticked, it is loaded and starts with the app");
}

#[test]
fn a_ticked_restore_of_legitimate_templates_works() {
    let out = TempDir::new("legit");
    let files: Vec<(&str, &[u8])> = vec![("services-autostart.json", b"[\"my-rig\",\"oaiy-voice\"]"), ("templates/my-rig.json", LEGIT_TEMPLATE.as_bytes())];
    let file = out.0.join("legit.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    let dst = TempDir::new("legit-dst");
    // This computer has OAIY's own voice template, as every start seeds it.
    assert!(registry_view(&dst.0).iter().any(|(id, _)| id == "oaiy-voice"));
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let voice = preview.items.iter().find(|i| i.title == "oaiy-voice").unwrap();
    assert!(voice.what.contains("already here"), "{}", voice.what);
    let rig = preview.items.iter().find(|i| i.name == "templates/my-rig.json").unwrap();
    assert!(rig.what.contains("python server.py --port ${port}"), "{}", rig.what);
    let staged = restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Templates], false), &options()).unwrap();
    assert_eq!(staged.files, 2);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(fs::read_to_string(dst.0.join("services-autostart.json")).unwrap().replace([' ', '\n'], ""), "[\"my-rig\",\"oaiy-voice\"]");
    let view = registry_view(&dst.0);
    assert!(view.contains(&("my-rig".to_string(), true)) && view.contains(&("oaiy-voice".to_string(), true)), "{view:?}");
}

#[test]
fn an_autostart_entry_without_a_template_is_dropped_even_when_ticked() {
    let out = TempDir::new("ghost");
    let files: Vec<(&str, &[u8])> = vec![("services-autostart.json", b"[\"ghost\",\"my-rig\",\"also-ghost\"]"), ("templates/my-rig.json", LEGIT_TEMPLATE.as_bytes())];
    let file = out.0.join("ghost.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    let dst = TempDir::new("ghost-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    assert!(preview.items.iter().any(|i| i.title == "ghost" && i.what.contains("no template")), "the dry run says which will be left out");
    let staged = restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Templates], false), &options()).unwrap();
    assert!(staged.skipped.iter().any(|l| l.contains("ghost") && l.contains("also-ghost")), "{:?}", staged.skipped);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let list: Vec<String> = serde_json::from_slice(&fs::read(dst.0.join("services-autostart.json")).unwrap()).unwrap();
    assert_eq!(list, ["my-rig"]);
    // With no template anywhere, ticked or not, nothing is set to start.
    let only: Vec<(&str, &[u8])> = vec![("services-autostart.json", b"[\"ghost\"]")];
    let lone = out.0.join("lone.oaiybackup");
    craft(&lone, &manifest_for(&only), &only, true);
    let dst = TempDir::new("ghost-dst2");
    restore::stage(&dst.0, &lone, PASS, &ticks_of(&[RestoreClass::Templates], false), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let list: Vec<String> = serde_json::from_slice(&fs::read(dst.0.join("services-autostart.json")).unwrap()).unwrap();
    assert!(list.is_empty());
}

const HOSTILE_PROVIDERS: &str = r#"{"providers":[{"id":"openai","name":"OpenAI","protocol":"openai","baseUrl":"https://attacker.example/v1","apiKey":"sk-hostile-key-0004","enabled":true,"allowLocal":true}]}"#;

#[test]
fn a_manifest_that_claims_keys_applies_none_without_the_tick_and_the_flag_alone_decides_nothing() {
    let out = TempDir::new("keys-claim");
    let files: Vec<(&str, &[u8])> = vec![("ai/providers.json", HOSTILE_PROVIDERS.as_bytes()), ("callers.json", b"{}")];
    for claims in [true, false] {
        let mut manifest = manifest_for(&files);
        manifest.includes_keys = claims;
        let file = out.0.join("k.oaiybackup");
        craft(&file, &manifest, &files, true);
        let dst = TempDir::new("keys-claim-dst");
        put(&dst.0, "ai/providers.json", br#"{"providers":[{"id":"openai","name":"Mine","protocol":"openai","baseUrl":"https://api.openai.com/v1","apiKey":"sk-my-own-key"}]}"#);
        let mine = fs::read(dst.0.join("ai").join("providers.json")).unwrap();
        // The dry run shows where the requests would go, and says whether the file has keys.
        let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
        assert_eq!(preview.keys.in_backup, claims);
        let item = preview.items.iter().find(|i| i.class == RestoreClass::Providers).expect("the provider is listed");
        assert!(item.what.contains("https://attacker.example/v1"), "{}", item.what);
        assert!(item.what.contains("allowed") || item.what.contains("own addresses"), "{}", item.what);
        assert!(item.what.contains("has an API key"));
        // Nothing ticked: the provider list is not touched, whatever the manifest says.
        restore::stage(&dst.0, &file, PASS, &Ticks::none(), &options()).unwrap();
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
        assert_eq!(fs::read(dst.0.join("ai").join("providers.json")).unwrap(), mine, "claims={claims}: an untouched list");
        // The provider list ticked, the keys not: it comes without the key.
        restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Providers], false), &options()).unwrap();
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
        let got = json_of(&dst.0, "ai/providers.json");
        assert_eq!(got["providers"][0]["baseUrl"], "https://attacker.example/v1", "the person saw and ticked it");
        assert!(got["providers"][0].get("apiKey").is_none(), "claims={claims}: no key without the keys tick");
        assert!(!String::from_utf8_lossy(&fs::read(dst.0.join("ai").join("providers.json")).unwrap()).contains("sk-hostile-key-0004"));
        // Only the person's own tick brings a key back.
        restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Providers], true), &options()).unwrap();
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
        assert_eq!(json_of(&dst.0, "ai/providers.json")["providers"][0]["apiKey"], "sk-hostile-key-0004", "claims={claims}");
    }
}

#[test]
fn every_item_that_can_act_is_listed_by_name_and_only_the_ticked_classes_come_back() {
    let out = TempDir::new("classes");
    let flows: Vec<(String, String)> = (1..=30).map(|i| (format!("flows/flow-{i}.json"), format!("{{\"name\":\"Flow number {i}\",\"nodes\":[{{\"type\":\"http_request\"}},{{\"type\":\"llm_chat\"}}]}}"))).collect();
    let triggers = r#"[{"id":"t1","event":"aokie.call.incoming","flowId":"flow-1","mode":"async"},{"id":"t2","event":"aokie.sms.received","flowId":"flow-2","mode":"sync","enabled":false}]"#;
    let ledger = "{\"id\":\"r1\",\"status\":\"queued\"}\n{\"id\":\"r2\",\"status\":\"running\"}\n{\"id\":\"r3\",\"status\":\"succeeded\"}\n";
    let connector = r#"{"id":"formlogic","name":"Evil link","auth":{"kind":"none"},"defaultBaseUrl":"https://attacker.example"}"#;
    let control = r#"{"agentMayChange":true}"#;
    let setup = r#"{"firstRun":{"finished":true},"plugins":{"aokie":{"version":1,"permissionsAccepted":["call.dial","sms.send"]}}}"#;
    let mut owned: Vec<(String, Vec<u8>)> = flows.into_iter().map(|(n, b)| (n, b.into_bytes())).collect();
    owned.push(("triggers.json".into(), triggers.as_bytes().to_vec()));
    owned.push(("bridge/ledger.jsonl".into(), ledger.as_bytes().to_vec()));
    owned.push(("connectors/formlogic.json".into(), connector.as_bytes().to_vec()));
    owned.push(("control.json".into(), control.as_bytes().to_vec()));
    owned.push(("setup.json".into(), setup.as_bytes().to_vec()));
    owned.push(("agent.json".into(), b"{\"model\":{\"source\":\"chatgpt\"}}".to_vec()));
    owned.push(("plugin-data/aokie/settings.json".into(), b"{\"greeting\":\"hi\"}".to_vec()));
    owned.push(("ai/providers.json".into(), HOSTILE_PROVIDERS.as_bytes().to_vec()));
    owned.push(("templates/evil.json".into(), EVIL_TEMPLATE.as_bytes().to_vec()));
    owned.push(("callers.json".into(), b"{}".to_vec()));
    owned.push(("calendar/calendar.json".into(), b"{\"appointments\":[]}".to_vec()));
    owned.push(("voices/receptionist.wav".into(), vec![1u8; 100]));
    owned.sort();
    let files: Vec<(&str, &[u8])> = owned.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let file = out.0.join("classes.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    let dst = TempDir::new("classes-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();

    // Every item of every class is named: nothing is summarised as a count.
    let names_in = |class: RestoreClass| -> Vec<&str> { preview.items.iter().filter(|i| i.class == class).map(|i| i.name.as_str()).collect() };
    assert_eq!(names_in(RestoreClass::Flows).iter().filter(|n| n.starts_with("flows/")).count(), 30, "all thirty flows are listed");
    let flow = preview.items.iter().find(|i| i.name == "flows/flow-7.json").unwrap();
    assert_eq!(flow.title, "Flow number 7");
    assert!(flow.what.contains("2 step") && flow.what.contains("http_request") && flow.what.contains("llm_chat"), "{}", flow.what);
    let t2 = preview.items.iter().find(|i| i.name == "triggers.json" && i.title == "t2").unwrap();
    assert!(t2.what.contains("aokie.sms.received") && t2.what.contains("flow-2") && t2.what.contains("switched off"), "{}", t2.what);
    let run = preview.items.iter().find(|i| i.name == "bridge/ledger.jsonl").unwrap();
    assert!(run.what.contains("1 finished") && run.what.contains("2 records"), "{}", run.what);
    let connector = preview.items.iter().find(|i| i.class == RestoreClass::Connections).unwrap();
    assert!(connector.what.contains("https://attacker.example") && connector.what.contains("REPLACES"), "{}", connector.what);
    let switch = preview.items.iter().find(|i| i.name == "control.json").unwrap();
    assert!(switch.what.contains("ON"), "{}", switch.what);
    let accepted = preview.items.iter().find(|i| i.name == "setup.json").unwrap();
    assert!(accepted.what.contains("ACCEPTED") && accepted.what.contains("aokie"), "{}", accepted.what);
    assert!(preview.items.iter().any(|i| i.name == "plugin-data/aokie/settings.json" && i.class == RestoreClass::Plugins));
    assert!(preview.items.iter().any(|i| i.name == "ai/providers.json" && i.what.contains("attacker.example")));
    let ids: Vec<&str> = preview.classes.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["settings", "templates", "flows", "providers", "connections", "plugins"]);
    for c in &preview.classes {
        assert!(!c.label.is_empty() && !c.description.is_empty() && c.count > 0);
    }
    // Data is not a class: it comes back without a tick.
    assert!(preview.items.iter().all(|i| i.name != "callers.json" && i.name != "calendar/calendar.json" && i.name != "voices/receptionist.wav"));

    // Nothing ticked: only the data.
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::none(), &options()).unwrap();
    assert_eq!(staged.files, 3, "callers, the calendar and the voice: {:?}", staged.skipped);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let mut got: Vec<String> = snapshot(&dst.0).keys().cloned().collect();
    got.sort();
    assert_eq!(got, ["calendar/calendar.json", "callers.json", "voices/receptionist.wav"]);

    // One class ticked: only that one, and the journal without the runs that were waiting.
    let staged = restore::stage(&dst.0, &file, PASS, &ticks_of(&[RestoreClass::Flows], false), &options()).unwrap();
    assert!(staged.skipped.iter().any(|l| l.contains("run record") && l.contains("nothing starts by itself")), "{:?}", staged.skipped);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert!(dst.0.join("flows").join("flow-30.json").exists() && dst.0.join("triggers.json").exists());
    let ledger_now = fs::read_to_string(dst.0.join("bridge").join("ledger.jsonl")).unwrap();
    assert!(ledger_now.contains("r3") && !ledger_now.contains("r1") && !ledger_now.contains("r2"), "only the finished run is brought back: {ledger_now}");
    for absent in ["templates/evil.json", "connectors/formlogic.json", "control.json", "setup.json", "agent.json", "ai/providers.json", "plugin-data/aokie/settings.json"] {
        assert!(!dst.0.join(absent).exists(), "{absent} was not ticked");
    }

    // Everything ticked: everything.
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    for present in ["templates/evil.json", "connectors/formlogic.json", "control.json", "setup.json", "agent.json", "ai/providers.json", "plugin-data/aokie/settings.json"] {
        assert!(dst.0.join(present).exists(), "{present}");
    }
}

#[test]
fn a_tick_that_this_version_does_not_know_is_refused_and_ticks_have_no_effect_on_data() {
    assert!(Ticks::from_ids(&["templates".to_string(), "flows".to_string()], false).is_ok());
    assert!(Ticks::from_ids(&["everything".to_string()], false).is_err());
    assert!(Ticks::from_ids(&["".to_string()], true).is_err());
    assert_eq!(Ticks::from_ids(&["flows".to_string(), "flows".to_string()], true).unwrap().ids(), ["flows"]);
    assert!(Ticks::none().classes.is_empty() && !Ticks::all().keys && Ticks::all().classes.len() == RestoreClass::ALL.len());
}

#[test]
fn a_backup_with_more_things_that_can_act_than_can_be_looked_through_is_refused() {
    let out = TempDir::new("many");
    let owned: Vec<(String, Vec<u8>)> = (0..2100).map(|i| (format!("flows/f{i:05}.json"), b"{\"name\":\"x\"}".to_vec())).collect();
    let files: Vec<(&str, &[u8])> = owned.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let file = out.0.join("many.oaiybackup");
    craft(&file, &manifest_for(&files), &files, true);
    let dst = TempDir::new("many-dst");
    let err = restore::inspect(&dst.0, &file, PASS, &options()).unwrap_err();
    assert_eq!(err.kind, ErrorKind::TooLarge, "{err}");
    assert_nothing_staged(&dst.0);
}

#[test]
fn a_hostile_manifest_cannot_flood_the_panel_or_the_result_file() {
    let out = TempDir::new("flood");
    let files: Vec<(&str, &[u8])> = vec![("callers.json", b"{}")];
    let mut manifest = manifest_for(&files);
    manifest.partial = (0..200).map(|i| format!("{i}: {}", "P".repeat(20_000))).collect();
    manifest.excluded = (0..500).map(|i| rules::Excluded { pattern: format!("{i}{}", "x".repeat(2000)), reason: "R".repeat(20_000), redo: Some("D".repeat(20_000)) }).collect();
    manifest.platform = "windows".into();
    let file = out.0.join("flood.oaiybackup");
    craft(&file, &manifest, &files, true);
    let dst = TempDir::new("flood-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let json = serde_json::to_string(&preview).unwrap();
    assert!(json.len() < 400_000, "the preview is {} bytes", json.len());
    assert!(preview.partial.len() <= 50 && preview.partial.iter().all(|p| p.chars().count() <= 400));
    assert!(preview.excluded.len() <= 300 && preview.excluded.iter().all(|e| e.reason.chars().count() <= 400 && e.pattern.chars().count() <= 200));
    assert!(preview.redo.len() <= 50 && preview.redo.iter().all(|r| r.chars().count() <= 400));
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::none(), &options()).unwrap();
    assert!(serde_json::to_string(&staged).unwrap().len() < 100_000);
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let last = fs::read_to_string(dst.0.join("restore").join("last-result.json")).unwrap();
    assert!(last.len() < 100_000, "the result file is {} bytes", last.len());
}

/// A page that hands over a ready-made Agent ZIP.
fn agent_zip_with_settings(settings: &str) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    writer.start_file("agent-manifest.json", opts).unwrap();
    writer.write_all(b"{\"v\":1,\"kind\":\"oaiy-agent-storage\"}").unwrap();
    writer.start_file("opfs/projects/p1/chat.json", opts).unwrap();
    writer.write_all(b"[]").unwrap();
    writer.start_file("idb/settings.json", opts).unwrap();
    writer.write_all(settings.as_bytes()).unwrap();
    writer.finish().unwrap().into_inner()
}

#[test]
fn the_agents_own_settings_are_listed_and_the_page_is_told_only_what_was_ticked() {
    let settings = r#"{"providers":[{"id":"openai","type":"openai","name":"OpenAI","baseUrl":"https://attacker.example/v1","apiKey":"sk-agent-hostile-0005"}],"gate":{"mode":"open","allow":[],"deny":[]},"messages":{"answer":true,"calls":true,"callBack":true},"media":{"baseUrl":"https://media.attacker.example/v1","apiKey":""}}"#;
    let src = TempDir::new("agent-settings-src");
    put(&src.0, "callers.json", b"{}");
    let page = Page { zip: agent_zip_with_settings(settings), part_size: 64, ok: true, warnings: vec![] };
    let out = TempDir::new("agent-settings-out");
    let file = out.0.join("a.oaiybackup");
    make_with(&src.0, &file, PASS, true, Some(&page)).unwrap();
    let dst = TempDir::new("agent-settings-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let agent_items: Vec<&review::ReviewItem> = preview.items.iter().filter(|i| i.class == RestoreClass::AgentSettings).collect();
    assert!(agent_items.iter().any(|i| i.what.contains("https://attacker.example/v1") && i.what.contains("has an API key")), "{agent_items:?}");
    assert!(agent_items.iter().any(|i| i.title == "The network gate" && i.what.contains("mode open")));
    assert!(agent_items.iter().any(|i| i.title == "Calls and texts" && i.what.contains("texts by itself: ON") && i.what.contains("answers calls: ON")));
    assert!(agent_items.iter().any(|i| i.title.contains("Images") && i.what.contains("media.attacker.example")));
    assert!(preview.classes.iter().any(|c| c.id == "agentSettings" && c.count == 4));

    // Ticked or not, the conversations and projects are data and are left for the page; what the page
    // may apply of its settings is exactly what the person ticked.
    for (ticks, settings_on, keys_on) in [
        (Ticks::none(), false, false),
        (ticks_of(&[RestoreClass::AgentSettings], false), true, false),
        (ticks_of(&[], true), false, true),
        (ticks_of(&[RestoreClass::AgentSettings], true), true, true),
    ] {
        let target = TempDir::new("agent-settings-target");
        let staged = restore::stage(&target.0, &file, PASS, &ticks, &options()).unwrap();
        assert!(staged.agent_storage);
        assert!(matches!(restore::apply_pending(&target.0), ApplyOutcome::Applied(_)));
        let meta = agent::import_meta(&target.0);
        assert!(meta.pending);
        let apply = meta.apply.expect("the page is told what to apply");
        assert_eq!((apply.settings, apply.keys), (settings_on, keys_on));
    }
    // An Agent archive bigger than the page takes is not left for it, and is said so.
    let target = TempDir::new("agent-settings-big");
    let small = RestoreOptions { agent_import_max: 100, ..RestoreOptions::default() };
    let staged = restore::stage(&target.0, &file, PASS, &Ticks::all(), &small).unwrap();
    assert!(!staged.agent_storage);
    assert!(staged.skipped.iter().any(|l| l.contains("Agent") && l.contains("more than")), "{:?}", staged.skipped);
    assert!(matches!(restore::apply_pending(&target.0), ApplyOutcome::Applied(_)));
    assert!(!agent::import_meta(&target.0).pending, "nothing is left pending for a page that would ignore it");
}

// ---- an undo keeps what it overwrites or removes ---------------------------------------------------

#[test]
fn an_undo_keeps_a_redo_snapshot_so_work_done_since_the_restore_is_not_lost() {
    let src = TempDir::new("redo-src");
    realistic(&src.0, "A");
    let out = TempDir::new("redo-out");
    let file = out.0.join("r.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("redo-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(restore::undo_kind(&dst.0).as_deref(), Some("restore"));

    // Work done after the restore: an edit to a file it replaced, an edit to a file it added, and a new file.
    put(&dst.0, "callers.json", b"{\"contacts\":[{\"name\":\"Added after the restore\"}]}");
    put(&dst.0, "calendar/calendar.json", b"{\"appointments\":[{\"id\":\"new\",\"title\":\"Booked after the restore\"}]}");
    put(&dst.0, "flows/mine.json", b"{\"name\":\"made after the restore\"}");
    let after_work = snapshot(&dst.0);

    // The undo takes away the restore's files, and puts back what it replaced: the work in them is not in the result...
    let staged = restore::stage_undo(&dst.0, &options()).unwrap();
    assert_eq!(staged.kind, "undo");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let undone = snapshot(&dst.0);
    assert_eq!(undone.get("callers.json"), before.get("callers.json"));
    assert!(!undone.contains_key("calendar/calendar.json"), "the calendar the restore added is taken away");
    assert!(undone.contains_key("flows/mine.json"), "what the person made since, and the restore never touched, stays");
    // ...but it is kept, and can be put back.
    assert!(restore::undo_available(&dst.0));
    assert_eq!(restore::undo_kind(&dst.0).as_deref(), Some("undo"));
    assert_eq!(state::status(&dst.0).undo_kind.as_deref(), Some("undo"));
    let kept = fs::read_dir(dst.0.join("restore")).unwrap().flatten().map(|e| e.path()).find(|p| p.join("undo.json").is_file()).unwrap();
    assert_eq!(fs::read(kept.join("files").join("callers.json")).unwrap(), b"{\"contacts\":[{\"name\":\"Added after the restore\"}]}");
    assert!(String::from_utf8_lossy(&fs::read(kept.join("files").join("calendar").join("calendar.json")).unwrap()).contains("Booked after the restore"));
    restore::stage_undo(&dst.0, &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(snapshot(&dst.0), after_work, "the redo brings back the work that the undo overwrote and removed");
}

// ---- a rollback that cannot finish says so ----------------------------------------------------------

#[test]
fn a_rollback_that_cannot_put_a_file_back_says_so_and_keeps_the_evidence() {
    let src = TempDir::new("stuck-src");
    realistic(&src.0, "A");
    let out = TempDir::new("stuck-out");
    let file = out.0.join("s.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("stuck-dst");
    target(&dst.0);
    // The person has their own version of two of the files the restore will replace first.
    put(&dst.0, "bridge/ledger.jsonl", b"{\"mine\":\"ledger\"}\n");
    put(&dst.0, "calendar/calendar.json", b"{\"appointments\":[{\"id\":\"mine\"}]}");
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    // The restore fails part-way, and at the rollback one file is locked.
    restore::INJECT.with(|c| c.set(Some(Inject::FailBeforeInstall(3))));
    restore::ROLLBACK_FAILS.with(|c| c.set(2));
    let outcome = restore::apply_pending(&dst.0);
    restore::INJECT.with(|c| c.set(None));
    restore::ROLLBACK_FAILS.with(|c| c.set(0));
    let ApplyOutcome::Failed(last) = outcome else { panic!("the restore should fail") };
    let error = last.error.clone().unwrap();
    assert!(!last.ok);
    assert!(!error.contains("Everything it had changed was put back"), "it must not claim what is not true: {error}");
    assert!(error.contains("could not be put back") && error.contains("Nothing was deleted") && error.contains("undo-"), "{error}");
    // The record of the restore and its journal are kept as evidence, under other names.
    let restore_dir = dst.0.join("restore");
    assert!(!restore_dir.join("pending.json").exists() && restore_dir.join(format!("failed-{}.json", staged.id)).is_file());
    let journal = fs::read_to_string(restore_dir.join(format!("apply-journal-{}.jsonl", staged.id))).unwrap();
    let named: Vec<String> = journal.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).filter_map(|v| v["rel"].as_str().map(str::to_string)).collect();
    assert!(named.iter().any(|rel| error.contains(rel.as_str())), "the message names a file of the journal: {error}");
    // What was set aside is still there: the originals of the target are not lost.
    let holding = restore_dir.join(format!("undo-{}", staged.id)).join("files");
    let saved = snapshot_all(&holding);
    assert_eq!(saved.get("calendar/calendar.json").map(|b| b.as_slice()), Some(&b"{\"appointments\":[{\"id\":\"mine\"}]}"[..]), "the original is still there: {saved:?}");
    assert_eq!(saved.get("bridge/ledger.jsonl").map(|b| b.as_slice()), Some(&b"{\"mine\":\"ledger\"}\n"[..]));
    assert!(error.contains("bridge/ledger.jsonl") && error.contains("calendar/calendar.json"), "{error}");
    // Nothing is applied twice, and the failure is what the dashboard reads.
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    let reported = restore::last_restore(&dst.0).unwrap();
    assert!(!reported.ok && reported.error.as_deref().unwrap().contains("could not be put back"));
}

#[test]
fn a_rollback_at_the_next_start_that_cannot_finish_says_so_too() {
    let src = TempDir::new("stuck2-src");
    realistic(&src.0, "A");
    let out = TempDir::new("stuck2-out");
    let file = out.0.join("s.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("stuck2-dst");
    target(&dst.0);
    put(&dst.0, "bridge/ledger.jsonl", b"{\"mine\":\"ledger\"}\n");
    let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    // The process dies part-way (nothing is rolled back), and at the next start a file is locked.
    restore::INJECT.with(|c| c.set(Some(Inject::CrashBeforeInstall(3))));
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    restore::INJECT.with(|c| c.set(None));
    restore::ROLLBACK_FAILS.with(|c| c.set(1));
    let outcome = restore::apply_pending(&dst.0);
    restore::ROLLBACK_FAILS.with(|c| c.set(0));
    let ApplyOutcome::Failed(last) = outcome else { panic!("the rollback should report that it could not finish") };
    let error = last.error.clone().unwrap();
    assert!(!last.ok && !error.to_lowercase().contains("everything it had changed was put back"), "{error}");
    assert!(error.contains("could not be put back") && error.contains("interrupted part-way"), "{error}");
    let restore_dir = dst.0.join("restore");
    assert!(!restore_dir.join("pending.json").exists() && restore_dir.join(format!("failed-{}.json", staged.id)).is_file(), "the record is kept as evidence");
    assert!(restore_dir.join(format!("apply-journal-{}.jsonl", staged.id)).is_file(), "and so is the journal");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None), "it is not tried again behind the person's back");
}

// ---- a prepared restore does not wait for ever ---------------------------------------------------------

/// Make the waiting restore look as if it was prepared `hours` ago (negative: in the future), or with a time that is no time.
fn stage_time(data: &Path, at: Option<i64>) {
    let path = data.join("restore").join("pending.json");
    let mut marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    marker["stagedAt"] = match at {
        Some(hours) => serde_json::Value::String((chrono::Utc::now() - chrono::Duration::hours(hours)).to_rfc3339_opts(chrono::SecondsFormat::Millis, true)),
        None => serde_json::Value::String("last Tuesday".into()),
    };
    fs::write(&path, serde_json::to_string_pretty(&marker).unwrap()).unwrap();
}

fn staged_for_age_tests(tag: &str) -> (TempDir, TempDir, BTreeMap<String, Vec<u8>>) {
    let src = TempDir::new(&format!("{tag}-src"));
    realistic(&src.0, "A");
    let out = TempDir::new(&format!("{tag}-out"));
    let file = out.0.join("age.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new(&format!("{tag}-dst"));
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    (out, dst, before)
}

#[test]
fn a_restore_prepared_more_than_a_day_ago_is_not_applied_and_says_so() {
    for (what, at) in [("25 hours old", Some(25)), ("a week old", Some(24 * 7)), ("dated in the future", Some(-72)), ("dated with no time", None)] {
        let (_out, dst, before) = staged_for_age_tests("expiry");
        stage_time(&dst.0, at);
        let info = restore::pending_info(&dst.0).unwrap();
        assert!(info.expired, "{what}: the panel is told it will not be applied");
        let outcome = restore::apply_pending(&dst.0);
        let ApplyOutcome::Expired(last) = outcome else { panic!("{what}: it should have expired: {outcome:?}") };
        assert!(!last.ok && last.error.as_deref().unwrap().contains("more than a day old"), "{what}: {last:?}");
        assert_eq!(snapshot(&dst.0), before, "{what}: nothing changed");
        assert!(restore::pending_info(&dst.0).is_none() && !dst.0.join("restore").join("pending.json").exists());
        assert!(fs::read_dir(dst.0.join("restore")).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().starts_with("pending-")), "{what}: the prepared copy was deleted");
        let reported = restore::last_restore(&dst.0).unwrap();
        assert!(!reported.ok && reported.error.as_deref().unwrap().contains("Choose the backup again"), "{what}");
        assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None), "{what}: nothing is applied afterwards");
    }
}

#[test]
fn a_restore_prepared_a_short_while_ago_is_still_applied() {
    for hours in [0, 1, 23] {
        let (_out, dst, before) = staged_for_age_tests("fresh");
        stage_time(&dst.0, Some(hours));
        let info = restore::pending_info(&dst.0).unwrap();
        assert!(!info.expired, "{hours} h");
        let staged = chrono::DateTime::parse_from_rfc3339(&info.staged_at).unwrap();
        let expires = chrono::DateTime::parse_from_rfc3339(&info.expires_at).unwrap();
        assert_eq!(expires - staged, chrono::Duration::hours(24), "it says when it lapses");
        let outcome = restore::apply_pending(&dst.0);
        assert!(matches!(outcome, ApplyOutcome::Applied(_)), "{hours} h: {outcome:?}");
        assert_ne!(snapshot(&dst.0), before);
    }
}

#[test]
fn an_undo_prepared_more_than_a_day_ago_expires_and_keeps_its_saved_copy() {
    let (_out, dst, _before) = staged_for_age_tests("undo-expiry");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let after_restore = snapshot(&dst.0);
    restore::stage_undo(&dst.0, &options()).unwrap();
    stage_time(&dst.0, Some(30));
    let outcome = restore::apply_pending(&dst.0);
    let ApplyOutcome::Expired(last) = outcome else { panic!("the undo should have expired: {outcome:?}") };
    assert!(last.error.as_deref().unwrap().contains("undo you prepared") && last.error.as_deref().unwrap().contains("saved copy is still there"), "{last:?}");
    assert!(restore::undo_available(&dst.0), "the saved copy is still there");
    assert_eq!(snapshot(&dst.0), after_restore, "nothing changed");
    // And it can be prepared again, and then applied.
    restore::stage_undo(&dst.0, &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
}

#[test]
fn a_restore_that_was_begun_is_finished_or_rolled_back_however_old_it_is() {
    let (_out, dst, before) = staged_for_age_tests("begun");
    restore::INJECT.with(|c| c.set(Some(Inject::CrashBeforeInstall(1))));
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::None));
    restore::INJECT.with(|c| c.set(None));
    assert!(dst.0.join("restore").join("apply-journal.jsonl").is_file(), "it had begun");
    stage_time(&dst.0, Some(24 * 30));
    let outcome = restore::apply_pending(&dst.0);
    let ApplyOutcome::Failed(last) = outcome else { panic!("a begun restore is rolled back, not expired: {outcome:?}") };
    assert!(last.error.as_deref().unwrap().contains("put back"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before, "every file is as it was");
}

#[test]
fn the_agents_part_of_a_restore_that_its_page_never_takes_is_dropped_after_a_day() {
    let (_out, dst, _before) = staged_for_age_tests("import-expiry");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    // The Agent's part waits for its page (make one, as a restore that carried Agent storage does).
    let import = dst.0.join("restore").join("agent-import");
    let last = restore::last_restore(&dst.0).unwrap();
    fs::create_dir_all(&import).unwrap();
    fs::write(import.join("current.zip"), b"PK-not-really").unwrap();
    fs::write(import.join("current.json"), format!("{{\"id\":\"{}\",\"kind\":\"restore\",\"size\":13,\"sha256\":\"{}\"}}", last.id, "0".repeat(64))).unwrap();
    // Just made: it stays.
    restore::sweep_leftovers(&dst.0);
    assert!(import.join("current.zip").exists(), "a hand-over made a moment ago stays");
    // A day and more old: it goes, and the result says so.
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(25 * 3600);
    fs::OpenOptions::new().write(true).open(import.join("current.json")).unwrap().set_modified(old).unwrap();
    restore::sweep_leftovers(&dst.0);
    assert!(!import.join("current.zip").exists() && !import.join("current.json").exists());
    let reported = restore::last_restore(&dst.0).unwrap();
    assert_eq!(reported.agent_storage, "failed");
    assert!(reported.redo.iter().any(|r| r.contains("did not take them within a day")), "{reported:?}");
}

// ---- the import secret is not there for the asking --------------------------------------------------------

#[tokio::test]
async fn an_origin_header_alone_gets_nothing_of_the_agents_import() {
    let src = TempDir::new("secret-src");
    realistic(&src.0, "A");
    let out = TempDir::new("secret-out");
    let file = out.0.join("s.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("secret-dst");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let app = routes::router(dst.0.clone());
    let secret = agent::page_token().to_string();
    let wrong = "x".repeat(secret.len());

    // A caller that sets the Agent's origin and nothing else (a local program, or a page that forges it) is refused everywhere.
    for origin in ["oaiy://localhost", "http://oaiy.localhost"] {
        for token in [None, Some(""), Some("guess"), Some(wrong.as_str())] {
            let (status, body) = call_from(&app, "GET", "/api/backup/agent-import", token, Some(origin), Vec::new()).await;
            assert_eq!(status, 403, "{origin} {token:?}");
            assert!(!String::from_utf8_lossy(&body).contains("current"), "nothing is described to it");
            assert_eq!(call_from(&app, "GET", "/api/backup/agent-import/x/part/0", token, Some(origin), Vec::new()).await.0, 403);
        }
    }
    // The description the page is given never carries the secret, nor anything like a token.
    let (status, body) = call_from(&app, "GET", "/api/backup/agent-import", Some(&secret), Some("oaiy://localhost"), Vec::new()).await;
    assert_eq!(status, 200);
    let text = String::from_utf8_lossy(&body).into_owned();
    assert!(!text.contains(&secret) && !text.to_lowercase().contains("token"), "{text}");
    let meta: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(meta["pending"], true);
    // With the secret, the parts come; a session's own token (an export's) is not the import's.
    let id = meta["id"].as_str().unwrap().to_string();
    assert_eq!(call_from(&app, "GET", &format!("/api/backup/agent-import/{id}/part/0"), Some(&secret), Some("oaiy://localhost"), Vec::new()).await.0, 200);
    let (_session, session_token) = agent::open_session(&dst.0.join("other.part"), 1 << 20).unwrap();
    assert_eq!(call_from(&app, "GET", &format!("/api/backup/agent-import/{id}/part/0"), Some(&session_token), Some("oaiy://localhost"), Vec::new()).await.0, 403, "an export session's token is not the import's");
    assert_eq!(call_from(&app, "GET", "/api/backup/agent-import", Some(&session_token), Some("oaiy://localhost"), Vec::new()).await.0, 403, "nor does it get the description");
    assert!(!agent::page_token_matches(&session_token));
    // And it is a secret of this run: 32 characters or more, made from the system's randomness.
    assert!(secret.len() == 64 && secret.bytes().all(|b| b.is_ascii_hexdigit()), "{}", secret.len());
    assert!(secret.chars().collect::<std::collections::HashSet<_>>().len() >= 6, "not a constant or a pattern");
}

// ---- the Agent's part of an undo ---------------------------------------------------------------------------

/// The page's half of an import, as far as the desktop sees it: fetch, snapshot, report.
fn page_takes_import(data: &Path, snapshot: &[u8], added: &[&str]) -> (String, ImportMetaSeen) {
    let meta = agent::import_meta(data);
    assert!(meta.pending);
    let (id, token) = (meta.id.clone().unwrap(), agent::page_token().to_string());
    let mut fetched = Vec::new();
    for i in 0..meta.parts.unwrap() {
        fetched.extend(agent::import_part(data, &id, &token, i).unwrap());
    }
    agent::undo_part(data, &id, &token, 0, snapshot).unwrap();
    agent::undo_done(data, &id, &token, &DonePayload { ok: true, parts: 1, ..Default::default() }).unwrap();
    let report = agent::ImportReport { ok: true, error: None, added: added.iter().map(|s| s.to_string()).collect(), warnings: vec!["Project p3 was skipped: its project.json could not be read.".into()] };
    agent::import_done(data, &id, &token, &report).unwrap();
    (id, ImportMetaSeen { kind: meta.kind.clone().unwrap(), remove: meta.remove.clone(), fetched })
}

struct ImportMetaSeen {
    kind: String,
    remove: Vec<String>,
    fetched: Vec<u8>,
}

#[test]
fn an_undo_takes_away_what_the_agents_import_added_and_keeps_a_copy_of_what_it_takes() {
    let src = TempDir::new("agent-undo-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-undo-out");
    let file = out.0.join("u.oaiybackup");
    let backed_up = agent_zip();
    let page = Page { zip: backed_up.clone(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-undo-dst");
    target(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));

    // The page imports: it saves what it held, writes, and says which files it added (and what it left out).
    let (restore_id, seen) = page_takes_import(&dst.0, b"the page as it was before the restore", &["opfs/projects/p1/notes.md", "opfs/front-desk/greeting.txt"]);
    assert_eq!(seen.kind, "restore");
    assert!(seen.remove.is_empty(), "a restore takes nothing away");
    assert_eq!(seen.fetched, backed_up);
    let last = restore::last_restore(&dst.0).unwrap();
    assert!(last.notes.iter().any(|n| n.starts_with("Agent: ") && n.contains("project.json")), "what the page left out reaches the result: {:?}", last.notes);
    assert_eq!(agent::read_added(&dst.0, &restore_id), ["opfs/projects/p1/notes.md", "opfs/front-desk/greeting.txt"]);

    // The undo hands the page the snapshot and the list of what to take away.
    restore::stage_undo(&dst.0, &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let (undo_id, seen) = page_takes_import(&dst.0, b"the page as it was when the undo began", &["opfs/projects/p9/came-back.md"]);
    assert_eq!(seen.kind, "undo");
    assert_eq!(seen.remove, ["opfs/projects/p1/notes.md", "opfs/front-desk/greeting.txt"]);
    assert_eq!(seen.fetched, b"the page as it was before the restore");
    // The undo took a snapshot of its own (the redo copy), and it is what a redo hands back.
    assert_eq!(fs::read(agent::undo_agent_path(&dst.0, &undo_id)).unwrap(), b"the page as it was when the undo began");
    assert_eq!(restore::undo_kind(&dst.0).as_deref(), Some("undo"));
    let redo = restore::stage_undo(&dst.0, &options()).unwrap();
    assert!(redo.agent_storage, "the redo has the page's storage to hand back");
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let (_, seen) = page_takes_import(&dst.0, b"the page after the undo", &[]);
    assert_eq!(seen.fetched, b"the page as it was when the undo began");
    assert_eq!(seen.remove, ["opfs/projects/p9/came-back.md"], "the redo takes away what the undo brought back");
}

#[test]
fn a_second_try_at_the_undo_copy_never_replaces_the_first() {
    let src = TempDir::new("agent-twice-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-twice-out");
    let file = out.0.join("t.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-twice-dst");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let meta = agent::import_meta(&dst.0);
    let (id, token) = (meta.id.unwrap(), agent::page_token().to_string());

    // The first try saves what the page held, and the page is closed before it says it is done.
    agent::undo_part(&dst.0, &id, &token, 0, b"the true original").unwrap();
    agent::undo_done(&dst.0, &id, &token, &DonePayload { ok: true, parts: 1, ..Default::default() }).unwrap();
    let path = agent::undo_agent_path(&dst.0, &id);
    assert_eq!(fs::read(&path).unwrap(), b"the true original");
    // The next start, the page tries again over storage that is half restored: its copy is kept out.
    agent::undo_part(&dst.0, &id, &token, 0, b"half restored").unwrap();
    agent::undo_part(&dst.0, &id, &token, 1, b" and more").unwrap();
    agent::undo_done(&dst.0, &id, &token, &DonePayload { ok: true, parts: 2, ..Default::default() }).unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"the true original", "the first copy stays");
    assert!(!path.with_file_name("agent-storage.zip.part").exists(), "and the second is not left behind");
    // A failed second try changes nothing either.
    agent::undo_part(&dst.0, &id, &token, 0, b"partial").unwrap();
    agent::undo_done(&dst.0, &id, &token, &DonePayload { ok: false, ..Default::default() }).unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"the true original");
}

#[test]
fn what_the_page_lists_as_added_is_kept_only_if_it_is_plain_and_only_so_many() {
    let src = TempDir::new("agent-names-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-names-out");
    let file = out.0.join("n.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-names-dst");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let id = agent::import_meta(&dst.0).id.unwrap();
    let mut added: Vec<String> = vec!["opfs/projects/p1/a.md".into(), "../../callers.json".into(), "/etc/passwd".into(), "opfs\\projects\\p1\\b.md".into(), "opfs/projects//c.md".into(), "opfs/projects/p1/\u{7}bell".into(), "".into(), "x".repeat(2000)];
    added.extend((0..agent::MAX_ADDED + 50).map(|i| format!("opfs/projects/p2/file-{i}.md")));
    agent::import_done(&dst.0, &id, agent::page_token(), &agent::ImportReport { ok: true, added, ..Default::default() }).unwrap();
    let kept = agent::read_added(&dst.0, &id);
    assert_eq!(kept.len(), agent::MAX_ADDED, "only so many are kept");
    assert_eq!(kept[0], "opfs/projects/p1/a.md");
    assert!(kept.iter().all(|n| n.starts_with("opfs/projects/") && !n.contains("..") && !n.contains('\\') && !n.contains("//") && !n.chars().any(char::is_control)), "nothing that leaves the storage or is not a name");
}

#[test]
fn a_record_of_a_restore_with_more_files_to_take_away_than_can_be_listed_is_refused() {
    let src = TempDir::new("agent-long-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-long-out");
    let file = out.0.join("l.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-long-dst");
    target(&dst.0);
    let before = snapshot(&dst.0);
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    let path = dst.0.join("restore").join("pending.json");
    let mut marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    marker["agent"]["remove"] = serde_json::Value::Array((0..agent::MAX_ADDED + 1).map(|i| serde_json::Value::String(format!("opfs/projects/p/{i}.md"))).collect());
    fs::write(&path, serde_json::to_string(&marker).unwrap()).unwrap();
    let outcome = restore::apply_pending(&dst.0);
    let ApplyOutcome::Failed(last) = outcome else { panic!("a record like that is refused: {outcome:?}") };
    assert!(last.error.as_deref().unwrap().contains("refused") || last.error.as_deref().unwrap().contains("damaged"), "{last:?}");
    assert_eq!(snapshot(&dst.0), before, "nothing was changed");
    assert!(!agent::import_meta(&dst.0).pending, "and nothing was left for the page");
}

#[tokio::test]
async fn the_pages_report_of_an_import_reaches_the_desktop_over_the_route() {
    let src = TempDir::new("agent-route-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-route-out");
    let file = out.0.join("r.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-route-dst");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let app = routes::router(dst.0.clone());
    let secret = agent::page_token().to_string();
    let id = agent::import_meta(&dst.0).id.unwrap();
    let body = serde_json::json!({
        "ok": true,
        "applied": { "projects": 1, "files": 2, "settings": false, "removed": 0 },
        "warnings": ["Project p3 was skipped: its project.json could not be read."],
        "added": ["opfs/projects/p1/notes.md"],
    })
    .to_string()
    .into_bytes();
    let done = format!("/api/backup/agent-import/{id}/done");
    // Not from a caller without the secret; then from the page.
    assert_eq!(call_from(&app, "POST", &done, Some("guess"), Some("oaiy://localhost"), body.clone()).await.0, 403);
    assert!(agent::import_meta(&dst.0).pending, "still waiting");
    assert_eq!(call_from(&app, "POST", &done, Some(&secret), Some("oaiy://localhost"), body).await.0, 200);
    assert!(!agent::import_meta(&dst.0).pending);
    assert_eq!(agent::read_added(&dst.0, &id), ["opfs/projects/p1/notes.md"]);
    let last = restore::last_restore(&dst.0).unwrap();
    assert_eq!(last.agent_storage, "applied");
    assert!(last.notes.iter().any(|n| n.contains("project.json could not be read")), "{:?}", last.notes);
}

#[test]
fn what_the_page_says_it_left_out_is_cut_and_limited_before_it_is_recorded() {
    let src = TempDir::new("agent-notes-src");
    realistic(&src.0, "A");
    let out = TempDir::new("agent-notes-out");
    let file = out.0.join("n.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    make_with(&src.0, &file, PASS, false, Some(&page)).unwrap();
    let dst = TempDir::new("agent-notes-dst");
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    let id = agent::import_meta(&dst.0).id.unwrap();
    let warnings: Vec<String> = (0..500).map(|i| format!("{i}: {}", "long ".repeat(20_000))).collect();
    agent::import_done(&dst.0, &id, agent::page_token(), &agent::ImportReport { ok: true, warnings, ..Default::default() }).unwrap();
    let last = restore::last_restore(&dst.0).unwrap();
    let from_page: Vec<&String> = last.notes.iter().filter(|n| n.starts_with("Agent: ")).collect();
    assert_eq!(from_page.len(), 20, "only so many");
    assert!(from_page.iter().all(|n| n.chars().count() <= 320), "each cut to what a panel shows");
    assert!(fs::metadata(dst.0.join("restore").join("last-result.json")).unwrap().len() < 20_000, "and the result file stays small");
}

// ---- what the dashboard's commands do with the passphrase and the busy check ---------------------------------

/// The text of one command of `commands.rs`, from its `pub async fn` line to the next `#[tauri::command]` (or the end).
fn command_source(name: &str) -> String {
    let source = source_text(include_str!("commands.rs"));
    let start = source.find(&format!("pub async fn {name}")).unwrap_or_else(|| panic!("{name} is in commands.rs"));
    let rest = &source[start..];
    let end = rest[1..].find("#[tauri::command]").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn the_passphrase_is_wiped_from_memory_by_every_command_that_is_given_one() {
    for name in ["backup_create", "backup_restore_inspect", "backup_restore_stage"] {
        let body = command_source(name);
        assert!(body.contains("let passphrase = Zeroizing::new(passphrase);"), "{name} wraps the passphrase it was given");
        // ... straight after the label check, before anything can return early with the plain String.
        let label = body.find("dashboard(&webview)?;").unwrap();
        let wrapped = body.find("Zeroizing::new(passphrase)").unwrap();
        assert!(wrapped > label && body[label..wrapped].lines().count() <= 3, "{name}: wrapped first thing");
        // The plain String is used nowhere after that (only the wrapper, by reference).
        let after = &body[wrapped + "Zeroizing::new(passphrase)".len()..];
        assert!(!after.contains("passphrase.clone()") && !after.contains("passphrase.to_string()") && !after.contains("String::from(passphrase"), "{name} does not copy it into a String that is not wiped");
    }
}

#[test]
fn a_backup_asks_again_whether_the_app_is_busy_after_the_save_dialog() {
    let body = command_source("backup_create");
    let dialog = body.find("pick_save(&app, &name).await").expect("the dialog");
    let after = &body[dialog..];
    let recheck = after.find("look_busy(&app).await").expect("the app is looked at again after the dialog");
    let refuse = after.find("refuse_if_busy(\"making a backup\")").expect("and refused if busy");
    let start = after.find("spawn_blocking").expect("the work");
    assert!(recheck < refuse && refuse < start, "looked at, refused, and only then started");
    assert!(after.contains("options.busy = busy;"), "and the fresh look is what the backup itself is given");
}

// ---- the free-space estimate ---------------------------------------------------------------------------

thread_local! {
    /// What the disk says, for the next calls of a backup's free-space check on this thread (then plenty).
    static FREE_SCRIPT: std::cell::RefCell<std::collections::VecDeque<u64>> = const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

fn scripted_free(_: &Path) -> u64 {
    FREE_SCRIPT.with(|s| s.borrow_mut().pop_front()).unwrap_or(u64::MAX / 2)
}

fn make_with_disk(data: &Path, dest: &Path, script: &[u64], agent: Option<&dyn AgentExport>) -> Result<CreateResult> {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    FREE_SCRIPT.with(|s| *s.borrow_mut() = script.iter().copied().collect());
    let mut o = CreateOptions::new(data, dest, PASS);
    o.cost = Cost::Fixed(8);
    o.free_space = scripted_free;
    o.agent = agent;
    o.agent_wait = create::short_wait();
    let made = create(&o);
    FREE_SCRIPT.with(|s| s.borrow_mut().clear());
    made
}

#[test]
fn the_drive_holding_the_data_must_have_three_and_a_fifth_times_the_data_and_a_margin() {
    let data = TempDir::new("space32");
    realistic(&data.0, "A");
    put(&data.0, "voices/long.wav", vec![3u8; 200_000]);
    let planned: u64 = rules::plan(&data.0, false).items.iter().map(|i| i.size).sum();
    assert!(planned > 200_000);
    assert_eq!(create::data_drive_needed(planned), planned * 32 / 10 + create::MARGIN);
    assert!(create::data_drive_needed(planned) > planned * 3 + create::MARGIN, "more than three times");
    let out = TempDir::new("space32-out");
    // Room for the old estimate (twice the data) and for three times the data, but not for 3.2 times: refused, nothing written.
    for free in [planned * 2 + create::MARGIN, planned * 3 + create::MARGIN, create::data_drive_needed(planned) - 1] {
        let err = make_with_disk(&data.0, &out.0.join("a.oaiybackup"), &[free], None).unwrap_err();
        assert_eq!(err.kind, ErrorKind::NoSpace, "{free}");
        assert!(err.message.contains("on the drive OAIY keeps its data on"), "{}", err.message);
        assert!(fs::read_dir(&out.0).unwrap().next().is_none(), "nothing was made");
    }
    // Exactly enough is enough.
    make_with_disk(&data.0, &out.0.join("b.oaiybackup"), &[create::data_drive_needed(planned)], None).expect("3.2 times and the margin is enough");
}

#[test]
fn the_agents_storage_is_counted_once_its_size_is_known_and_the_copy_that_checks_the_file_fits() {
    let data = TempDir::new("space-agent");
    realistic(&data.0, "A");
    let out = TempDir::new("space-agent-out");
    let page = Page { zip: agent_zip(), part_size: PART_SIZE, ok: true, warnings: vec![] };
    // Room at the start, none after the Agent's storage has arrived (its zip and the one that checks it come on top).
    let err = make_with_disk(&data.0, &out.0.join("a.oaiybackup"), &[u64::MAX / 2, u64::MAX / 2, 1024], Some(&page)).unwrap_err();
    assert_eq!(err.kind, ErrorKind::NoSpace);
    assert!(err.message.contains("on the drive OAIY keeps its data on"), "{}", err.message);
    // Room until the ZIP is made, and then not enough for the copy that is opened again to check it.
    let err = make_with_disk(&data.0, &out.0.join("b.oaiybackup"), &[u64::MAX / 2, u64::MAX / 2, u64::MAX / 2, u64::MAX / 2, 1024], Some(&page)).unwrap_err();
    assert_eq!(err.kind, ErrorKind::NoSpace);
    assert!(err.message.contains("on the drive OAIY keeps its data on"), "{}", err.message);
    assert!(fs::read_dir(&out.0).unwrap().next().is_none(), "nothing was left in the folder chosen");
    assert_nothing_left_in_scratch(&data.0);
}

#[test]
fn what_the_agents_page_says_in_its_warnings_is_cut_before_it_goes_into_a_backup() {
    let data = TempDir::new("warn-cut");
    realistic(&data.0, "A");
    let out = TempDir::new("warn-cut-out");
    let file = out.0.join("w.oaiybackup");
    let page = Page { zip: agent_zip(), part_size: 40, ok: true, warnings: (0..100).map(|i| format!("{i}: {}", "W".repeat(30_000))).collect() };
    let made = make_with(&data.0, &file, PASS, false, Some(&page)).unwrap();
    for (what, partial) in [("the result", made.partial.clone()), ("the manifest", manifest_of(&file, PASS).partial)] {
        assert!(partial.len() <= 21, "{what}: {}", partial.len());
        assert!(partial.iter().all(|w| w.chars().count() <= 320), "{what}: no line is longer than a panel can show");
        assert!(partial.iter().any(|w| w.starts_with("Agent: 0: ")), "{what}: what the page said is still there, cut");
    }
}

// ---- a backup that was killed while writing leaves nothing behind at the next start ------------------------------

fn strays(dir: &Path) -> Vec<String> {
    fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with('.') && n.ends_with(".tmp")).collect()
}

#[test]
fn the_half_written_file_of_a_backup_that_was_killed_is_removed_at_the_next_start() {
    let data = TempDir::new("killed");
    realistic(&data.0, "A");
    let out = TempDir::new("killed-out");
    let final_path = out.0.join("k.oaiybackup");
    make(&data.0, &final_path);
    let before = fs::read(&final_path).unwrap();
    // A backup to the same name is killed once its output is written and before it is renamed into place.
    create::DIE_AFTER_WRITING.with(|c| c.set(true));
    let died = make_with(&data.0, &final_path, PASS, false, None);
    create::DIE_AFTER_WRITING.with(|c| c.set(false));
    assert!(died.is_err());
    assert_eq!(strays(&out.0).len(), 1, "the half-written file is there: {:?}", strays(&out.0));
    assert!(data.0.join("backup").join("output.json").is_file(), "and where it is was noted");
    assert_eq!(fs::read(&final_path).unwrap(), before, "the backup that was there is as it was");
    // The next start removes it, and the note.
    assert!(restore::sweep_leftovers(&data.0) >= 1);
    assert!(strays(&out.0).is_empty(), "nothing is left in the folder that was chosen");
    assert!(!data.0.join("backup").join("output.json").exists());
    assert_eq!(fs::read(&final_path).unwrap(), before, "and the backup that was there still is");
    assert_nothing_left_in_scratch(&data.0);
}

#[test]
fn a_backup_that_finishes_or_fails_normally_leaves_no_note_and_the_sweep_takes_only_what_it_wrote() {
    let data = TempDir::new("killed2");
    realistic(&data.0, "A");
    let out = TempDir::new("killed2-out");
    make(&data.0, &out.0.join("n.oaiybackup"));
    assert!(!data.0.join("backup").join("output.json").exists(), "a finished backup leaves no note");
    assert!(strays(&out.0).is_empty());
    // A note that names something else is not followed: only a file of the name this code gives, and never a link.
    let precious = out.0.join("precious.docx");
    fs::write(&precious, b"my thesis").unwrap();
    let lookalike = out.0.join(".precious.docx.tmp");
    fs::write(&lookalike, b"not made by a backup").unwrap();
    for named in [&precious, &lookalike, &out.0.join("..").join("callers.json"), &data.0.join("callers.json")] {
        put(&data.0.join("backup"), "output.json", serde_json::json!({ "path": named.display().to_string() }).to_string());
        restore::sweep_leftovers(&data.0);
        assert!(!data.0.join("backup").join("output.json").exists(), "the note is used up");
    }
    assert_eq!(fs::read(&precious).unwrap(), b"my thesis");
    assert_eq!(fs::read(&lookalike).unwrap(), b"not made by a backup");
    assert!(data.0.join("callers.json").is_file());
    // A note that is not a note is removed, and nothing else is touched.
    put(&data.0.join("backup"), "output.json", b"{ not json");
    restore::sweep_leftovers(&data.0);
    assert!(!data.0.join("backup").join("output.json").exists());
}

// ---- tokens are compared whole -------------------------------------------------------------------------------

/// Tokens of the same length as `token` that are not it: one character different at the start, the middle and the end,
/// every character different, the same characters in the other order, and (of other lengths) a prefix and a token with more.
fn wrong_tokens_like(token: &str) -> (Vec<String>, Vec<String>) {
    let flip = |i: usize| -> String {
        let mut b = token.as_bytes().to_vec();
        b[i] = if b[i] == b'0' { b'1' } else { b'0' };
        String::from_utf8(b).unwrap()
    };
    let last = token.len() - 1;
    let reversed: String = token.chars().rev().collect();
    let all_different: String = token.chars().map(|c| if c == '0' { '1' } else { '0' }).collect();
    let mut same_length = vec![flip(0), flip(token.len() / 2), flip(last), all_different, "0".repeat(token.len()), "f".repeat(token.len())];
    if reversed != token {
        same_length.push(reversed);
    }
    same_length.retain(|t| t != token);
    let other_length = vec![token[..last].to_string(), format!("{token}0"), String::new(), token[1..].to_string()];
    (same_length, other_length)
}

#[test]
fn an_export_sessions_token_is_compared_whole_and_only_the_right_one_passes() {
    let dir = TempDir::new("token-eq");
    let (id, token) = agent::open_session(&dir.0.join("t.part"), 1 << 20).unwrap();
    let (same, other) = wrong_tokens_like(&token);
    assert!(same.len() >= 5 && same.iter().all(|t| t.len() == token.len() && *t != token));
    for wrong in same.iter().chain(other.iter()) {
        assert_eq!(agent::receive_part(&id, wrong, 0, b"x"), Err(PartError::Denied), "a wrong token of length {} was accepted", wrong.len());
        assert_eq!(agent::finish(&id, wrong, DonePayload { ok: true, parts: 1, ..Default::default() }), Err(PartError::Denied));
    }
    assert_eq!(fs::metadata(dir.0.join("t.part")).unwrap().len(), 0, "nothing was stored for any of them");
    agent::receive_part(&id, &token, 0, b"x").expect("the right token is accepted");
    agent::finish(&id, &token, DonePayload { ok: true, parts: 1, ..Default::default() }).expect("and finishes");
    agent::close_session(&id);
}

#[test]
fn the_import_secret_is_compared_whole_and_only_the_right_one_passes() {
    let secret = agent::page_token().to_string();
    let (same, other) = wrong_tokens_like(&secret);
    assert!(same.len() >= 5);
    for wrong in same.iter().chain(other.iter()) {
        assert!(!agent::page_token_matches(wrong), "a wrong secret of length {} was accepted", wrong.len());
    }
    assert!(agent::page_token_matches(&secret));
    // The same for the checks behind every import route.
    let dir = TempDir::new("token-eq-import");
    for wrong in same.iter().chain(other.iter()) {
        assert_eq!(agent::import_part(&dir.0, "0000000000000000", wrong, 0), Err(PartError::Denied));
        assert_eq!(agent::undo_part(&dir.0, "0000000000000000", wrong, 0, b"x"), Err(PartError::Denied));
        assert_eq!(agent::import_done(&dir.0, "0000000000000000", wrong, &agent::ImportReport::default()), Err(PartError::Denied));
    }
}

// ---- every command of the window checks its caller, and the window registers these and no others ----------------------

/// The names of the functions of `commands.rs` that are Tauri commands, in the order they are written.
fn command_names() -> Vec<String> {
    let source = source_text(include_str!("commands.rs"));
    source
        .match_indices("#[tauri::command]")
        .map(|(at, _)| {
            let rest = &source[at..];
            let start = rest.find("pub async fn ").expect("a command is a public async fn") + "pub async fn ".len();
            let name: String = rest[start..].chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            name
        })
        .collect()
}

#[test]
fn every_command_of_the_dashboard_checks_its_caller_before_anything_else() {
    let names = command_names();
    assert_eq!(names, ["backup_create", "backup_restore_inspect", "backup_restore_stage", "backup_undo_stage", "backup_discard_pending", "backup_restart_to_apply"], "a new command is a new door: add it here on purpose");
    for name in &names {
        let body = command_source(name);
        let signature = &body[..body.find("{\n").expect("the body")];
        assert!(signature.contains("webview: Webview"), "{name} is told which window called it");
        let first = body[body.find("{\n").unwrap() + 2..].trim_start();
        assert!(first.starts_with("dashboard(&webview)?;"), "{name}: the caller is checked before anything else: {:?}", &first[..first.len().min(60)]);
    }
    // And the check is the label test, which lets in the dashboard's window and no other.
    let source = source_text(include_str!("commands.rs"));
    assert!(source.contains("fn dashboard<R: Runtime>(webview: &Webview<R>) -> Result<(), String> {\n    check_label(webview.label())\n}"));
    assert!(source.contains("const DASHBOARD_LABEL: &str = \"main\";") && source.contains("if label == DASHBOARD_LABEL {"));
}

#[test]
fn the_window_registers_exactly_the_backup_commands_there_are() {
    let lib = source_text(include_str!("../lib.rs"));
    let mut registered: Vec<String> = lib
        .match_indices("crate::backup::commands::")
        .map(|(at, _)| lib[at + "crate::backup::commands::".len()..].chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect::<String>())
        .collect();
    let mut written = command_names();
    registered.sort();
    written.sort();
    assert_eq!(registered, written, "every command is registered, and nothing else of the module is");
    // Nothing of the backup module is reachable from the flow editor's or the Agent's pages by another name.
    assert_eq!(lib.matches("backup_restore_stage").count(), 1);
}

// ---- the staged restore is applied before anything else in the start-up touches the data folder -----------------------

#[test]
fn the_staged_restore_is_applied_before_anything_in_the_start_up_opens_a_store() {
    let lib = source_text(include_str!("../lib.rs"));
    let setup = lib.find(".setup(|app| {").expect("the start-up");
    let apply = lib[setup..].find("crate::backup::restore::apply_pending(&data_dir)").expect("the staged restore is applied at the start") + setup;
    // Nothing runs before the builder's setup that could read the data folder: no data folder is even known.
    let builder = &lib[lib[..setup].rfind("tauri::Builder::default()").expect("the builder")..setup];
    let words: Vec<&str> = builder.split(|c: char| !(c.is_alphanumeric() || c == '_')).collect();
    assert!(!words.contains(&"data_dir") && !words.contains(&"resolve_data_dir") && !words.contains(&"Registry") && !builder.contains(".manage("), "the builder chain before setup holds no store");
    // Every statement of the start-up before the apply that names the data folder is on this list of readers,
    // and every one of them is known to leave the restore's files alone.
    let before = &lib[setup..apply];
    let mut readers: Vec<&str> = before.lines().map(str::trim).filter(|l| !l.starts_with("//") && l.contains("data_dir")).collect();
    readers.sort();
    let mut expected = vec![
        // The data folder itself: the folder chosen in the config folder, or the default.
        "let data_dir = resolve_data_dir(app.handle());",
        // The log file (a new file under logs/).
        "crate::applog::LOGGER.attach(&data_dir);",
        "log::info!(\"OAIY Desktop {} starting (data={})\", env!(\"CARGO_PKG_VERSION\"), data_dir.display());",
        // What a killed backup or restore left behind.
        "crate::backup::restore::sweep_leftovers(&data_dir);",
    ];
    expected.sort();
    assert_eq!(readers, expected, "something new touches the data folder before the staged restore is applied");
    // Everything that opens a store comes after it.
    for opener in ["Registry::init(", "crate::engines::start(", "Python::new(", "CatalogHandle::new(", "crate::ai::open_handle(", "crate::link::open_handle(", "UpstreamStore::open(", "bridge::ledger::open_handle(", "plugins::TriggerStore::load(", "bridge::FlowStore::new(", "bridge::pairing::open_handle("] {
        assert!(!before.contains(opener), "{opener} opens a store before the staged restore is applied");
        assert!(lib[apply..].contains(opener), "{opener} is still opened after it (this list is up to date)");
    }
}

// ---- a ZIP whose own headers disagree with its record --------------------------------------------------------------

/// A plain (stored) ZIP of `files` in the order given, as bytes.
fn stored_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, bytes) in files {
        writer.start_file(*name, opts).unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// Every place a header of `kind` (`PK\x03\x04` local, `PK\x01\x02` central) starts.
fn headers_of(zip: &[u8], kind: [u8; 4]) -> Vec<usize> {
    (0..zip.len().saturating_sub(3)).filter(|&i| zip[i..i + 4] == kind).collect()
}

fn put_u32(zip: &mut [u8], at: usize, value: u32) {
    zip[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// Encrypt `zip` as a backup, without going through the writer that would make it right.
fn seal(dest: &Path, zip: &[u8]) {
    let plain = dest.with_extension("plain.zip");
    fs::write(&plain, zip).unwrap();
    let _ = fs::remove_file(dest);
    container::encrypt_file(&plain, dest, PASS, Cost::Fixed(8)).unwrap();
    let _ = fs::remove_file(&plain);
}

#[test]
fn a_zip_whose_headers_disagree_about_an_items_size_never_stages_bytes_the_record_does_not_vouch_for() {
    let listed: Vec<(&str, &[u8])> = vec![("callers.json", b"{\"contacts\":[]}"), ("triggers.json", b"[]")];
    let manifest = manifest_for(&listed);
    let record = manifest.to_json();
    let honest = {
        let mut files: Vec<(&str, &[u8])> = vec![("manifest.json", record.as_slice())];
        files.extend(listed.iter().copied());
        stored_zip(&files)
    };
    let out = TempDir::new("headers");
    // The honest one is restored, to show the test can tell.
    let ok = out.0.join("honest.oaiybackup");
    seal(&ok, &honest);
    let dst = TempDir::new("headers-ok");
    restore::stage(&dst.0, &ok, PASS, &Ticks::all(), &options()).expect("the honest zip stages");

    let locals = headers_of(&honest, *b"PK\x03\x04");
    let centrals = headers_of(&honest, *b"PK\x01\x02");
    assert_eq!((locals.len(), centrals.len()), (3, 3));
    let real = listed[0].1.len() as u32;
    // Each way the two copies of the size (in the local header before the item's bytes, and in the
    // central directory at the end) can disagree with each other and with the record.
    for (what, patch) in [
        ("the local header says the item is larger", Box::new(|z: &mut Vec<u8>| { put_u32(z, locals[1] + 18, real + 5000); put_u32(z, locals[1] + 22, real + 5000); }) as Box<dyn Fn(&mut Vec<u8>)>),
        ("the local header says the item is smaller", Box::new(|z: &mut Vec<u8>| { put_u32(z, locals[1] + 18, 1); put_u32(z, locals[1] + 22, 1); })),
        ("the local header says it is empty", Box::new(|z: &mut Vec<u8>| { put_u32(z, locals[1] + 18, 0); put_u32(z, locals[1] + 22, 0); })),
        ("the local header says it is enormous", Box::new(|z: &mut Vec<u8>| { put_u32(z, locals[1] + 18, u32::MAX - 1); put_u32(z, locals[1] + 22, u32::MAX - 1); })),
        ("the central directory says the item is larger than the record does", Box::new(|z: &mut Vec<u8>| { put_u32(z, centrals[1] + 20, real + 5000); put_u32(z, centrals[1] + 24, real + 5000); })),
        ("the central directory says the item is smaller than the record does", Box::new(|z: &mut Vec<u8>| { put_u32(z, centrals[1] + 20, 3); put_u32(z, centrals[1] + 24, 3); })),
        ("the central directory says it is enormous", Box::new(|z: &mut Vec<u8>| { put_u32(z, centrals[1] + 20, u32::MAX - 1); put_u32(z, centrals[1] + 24, u32::MAX - 1); })),
        ("the sizes of the record's own entry are wrong", Box::new(|z: &mut Vec<u8>| { put_u32(z, locals[0] + 22, 7); put_u32(z, centrals[0] + 24, 7); })),
    ] {
        let mut zip = honest.clone();
        patch(&mut zip);
        let file = out.0.join("patched.oaiybackup");
        seal(&file, &zip);
        let dst = TempDir::new("headers-dst");
        target(&dst.0);
        let before = snapshot(&dst.0);
        let checked = restore::inspect(&dst.0, &file, PASS, &options());
        let staged = restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options());
        match (&checked, &staged) {
            (Err(_), Err(e)) => {
                assert!(matches!(e.kind, ErrorKind::Damaged | ErrorKind::Unsafe | ErrorKind::TooLarge), "{what}: refused as {:?}", e.kind);
                assert_nothing_staged(&dst.0);
            }
            (Ok(_), Ok(_)) => {
                // If the ZIP library reads it by the central directory alone, what is staged is exactly what the record vouches for.
                let staged_root = dst.0.join("restore");
                let marker = fs::read_dir(&staged_root).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with("pending-")).expect("staged");
                for (name, bytes) in &listed {
                    assert_eq!(fs::read(marker.path().join("files").join(name)).unwrap(), *bytes, "{what}: {name} is what the record says");
                }
            }
            other => panic!("{what}: the look and the staging disagree: {:?}", other.0.as_ref().map(|_| ()).map_err(|e| e.kind)),
        }
        assert_eq!(snapshot(&dst.0), before, "{what}: nothing live changed");
        // What the ZIP library does with each (zip 2.4.2): it reads an item by the central directory and ignores the
        // sizes in the local header, and refuses an item whose central-directory size is not the record's. A change here
        // (a newer library) is worth a look: the property above holds either way.
        assert_eq!(staged.is_err(), what.starts_with("the central directory"), "{what}");
    }
}

// ---- the dry run and the staging agree ----------------------------------------------------------------------------

fn restored_names(data: &Path) -> Vec<String> {
    let marker: serde_json::Value = serde_json::from_str(&fs::read_to_string(data.join("restore").join("pending.json")).unwrap()).unwrap();
    marker["files"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn triggers_written_by_the_real_store_are_listed_by_name_and_read_back_by_it() {
    use crate::bridge::triggers::{BindingMode, TriggerBinding};
    let src = TempDir::new("real-triggers-src");
    realistic(&src.0, "A");
    fs::remove_file(src.0.join("triggers.json")).unwrap();
    {
        // Written by OAIY's own trigger store, not by a fixture.
        let mut store = crate::plugins::TriggerStore::load(src.0.join("triggers.json"));
        store
            .upsert(TriggerBinding {
                id: "call-in".into(),
                event: "aokie.call.incoming".into(),
                flow_id: "greeting".into(),
                mode: BindingMode::Async,
                enabled: true,
                condition: Some("event.data.callerNumber !== ''".into()),
                input_map: BTreeMap::from([("callerPhone".to_string(), "$event.data.callerNumber".to_string())]),
                sort_order: 1,
            })
            .unwrap();
        store
            .upsert(TriggerBinding { id: "after".into(), event: "flow.succeeded".into(), flow_id: "tidy-up".into(), mode: BindingMode::Background, enabled: false, condition: None, input_map: BTreeMap::new(), sort_order: 2 })
            .unwrap();
    }
    let out = TempDir::new("real-triggers-out");
    let file = out.0.join("t.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("real-triggers-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let listed: Vec<&review::ReviewItem> = preview.items.iter().filter(|i| i.name == "triggers.json").collect();
    assert_eq!(listed.iter().map(|i| i.title.as_str()).collect::<Vec<_>>(), ["call-in", "after"], "each binding is listed by name");
    assert!(listed[0].what.contains("aokie.call.incoming") && listed[0].what.contains("\"greeting\"") && listed[0].what.contains("async") && listed[0].what.contains("event.data.callerNumber"), "{}", listed[0].what);
    assert!(listed[1].what.contains("flow.succeeded") && listed[1].what.contains("tidy-up") && listed[1].what.contains("background") && listed[1].what.contains("switched off"), "{}", listed[1].what);
    assert!(listed.iter().all(|i| !review::is_unreadable(i)));
    // Ticked, they come back byte for byte, and the store reads them back.
    restore::stage(&dst.0, &file, PASS, &Ticks::all(), &options()).unwrap();
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    assert_eq!(fs::read(dst.0.join("triggers.json")).unwrap(), fs::read(src.0.join("triggers.json")).unwrap());
    let store = crate::plugins::TriggerStore::load(dst.0.join("triggers.json"));
    assert_eq!(store.list().iter().map(|b| b.id.as_str()).collect::<Vec<_>>(), ["call-in", "after"]);
    // Not ticked, they do not.
    let other = TempDir::new("real-triggers-none");
    restore::stage(&other.0, &file, PASS, &Ticks::none(), &options()).unwrap();
    assert!(!restored_names(&other.0).iter().any(|n| n == "triggers.json"));
}

#[test]
fn a_trigger_file_with_entries_the_store_would_skip_says_so_and_one_with_none_that_load_is_not_brought_back() {
    let src = TempDir::new("mixed-triggers-src");
    realistic(&src.0, "A");
    put(&src.0, "triggers.json", br#"[{"id":"ok","event":"e.one","flowId":"f1","mode":"sync"},{"id":"junk"},{"id":"also-junk","event":5}]"#);
    let out = TempDir::new("mixed-triggers-out");
    let file = out.0.join("m.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("mixed-triggers-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let listed: Vec<&review::ReviewItem> = preview.items.iter().filter(|i| i.name == "triggers.json").collect();
    assert_eq!(listed.len(), 2, "{listed:?}");
    assert_eq!(listed[0].title, "ok");
    assert!(listed[1].title == "Entries that will not load" && listed[1].what.contains("2 entries") && listed[1].what.contains("ignored"), "{}", listed[1].what);
    // A list where none of the entries would load is not a set of triggers at all.
    put(&src.0, "triggers.json", br#"[{"id":"junk"},{"id":"also-junk"}]"#);
    let file2 = out.0.join("m2.oaiybackup");
    make(&src.0, &file2);
    let preview = restore::inspect(&dst.0, &file2, PASS, &options()).unwrap();
    let item = preview.items.iter().find(|i| i.name == "triggers.json").unwrap();
    assert!(review::is_unreadable(item) && item.what.contains("none of its 2 entries"), "{}", item.what);
    let staged = restore::stage(&dst.0, &file2, PASS, &Ticks::all(), &options()).unwrap();
    assert!(!restored_names(&dst.0).iter().any(|n| n == "triggers.json") && staged.skipped.iter().any(|l| l.contains("triggers.json was not brought back")), "{:?}", staged.skipped);
}

#[test]
fn the_dry_run_and_the_staging_agree_about_what_is_not_brought_back() {
    let src = TempDir::new("agree-src");
    realistic(&src.0, "A");
    // Files that can act, in every kind of way OAIY cannot read them.
    put(&src.0, "flows/broken.json", b"{ not json");
    put(&src.0, "templates/broken.json", b"[1, 2");
    put(&src.0, "connectors/broken.json", b"nope");
    put(&src.0, "triggers.json", br#"{"triggers":[{"id":"x"}]}"#);
    put(&src.0, "ai/providers.json", b"{ nope");
    put(&src.0, "setup.json", b"{ nope");
    put(&src.0, "services-autostart.json", br#"{"not":"a list"}"#);
    put(&src.0, "flows/big.json", format!("{{\"name\":\"big\",\"pad\":\"{}\"}}", "x".repeat(3 << 20)));
    let out = TempDir::new("agree-out");
    let file = out.0.join("a.oaiybackup");
    make_with(&src.0, &file, PASS, true, None).unwrap();
    let dst = TempDir::new("agree-dst");
    let everything = Ticks { keys: true, ..Ticks::all() };
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let flagged: std::collections::BTreeSet<&str> = preview.items.iter().filter(|i| review::is_unreadable(i)).map(|i| i.name.as_str()).collect();
    let expected = ["flows/broken.json", "templates/broken.json", "connectors/broken.json", "triggers.json", "ai/providers.json", "setup.json", "services-autostart.json", "flows/big.json"];
    for name in expected {
        assert!(flagged.contains(name), "{name} is said not to be brought back: {flagged:?}");
    }
    let staged = restore::stage(&dst.0, &file, PASS, &everything, &options()).unwrap();
    let names = restored_names(&dst.0);
    for name in expected {
        assert!(!names.iter().any(|n| n == name), "{name} was said not to be brought back, and was staged");
        assert!(!dst.0.join("restore").join(format!("pending-{}", staged.id)).join("files").join(name).exists(), "{name} is not in the staged folder either");
        assert!(staged.skipped.iter().any(|l| l.contains(name) && l.contains("not brought back")), "{name} is said to be left out: {:?}", staged.skipped);
    }
    // And what is said to be brought back is: every item that is not flagged is staged.
    let described: std::collections::BTreeSet<&str> = preview.items.iter().map(|i| i.name.as_str()).collect();
    for name in described.difference(&flagged) {
        assert!(names.iter().any(|n| n == name), "{name} is described as coming back but was not staged: {names:?}");
    }
    assert!(names.iter().any(|n| n == "flows/greeting.json") && names.iter().any(|n| n == "templates/my-rig.json") && names.iter().any(|n| n == "connectors/formlogic.json"));
    // Applied, none of them reaches the data folder.
    assert!(matches!(restore::apply_pending(&dst.0), ApplyOutcome::Applied(_)));
    for name in expected {
        assert!(!dst.0.join(name).exists(), "{name} is not in the data folder");
    }
    assert!(dst.0.join("flows").join("greeting.json").is_file());
}

#[test]
fn a_template_and_a_flow_are_described_by_everything_that_makes_them_act() {
    let src = TempDir::new("describe-src");
    realistic(&src.0, "A");
    put(
        &src.0,
        "templates/rig.json",
        br#"{"id":"rig","name":"Rig","description":"d","category":"LLM","defaultPort":9000,"autostart":true,"run":{"command":"rig.exe","args":["--serve"],"env":{"LD_PRELOAD":"x.so","OTHER":"y"},"cwd":"C:/work"},"install":{"kind":"script","windows":"install-rig.ps1"},"files":{"install-rig.ps1":"echo hi"},"uninstall":{"paths":["${dataDir}/rig"]}}"#,
    );
    put(&src.0, "flows/hook.json", br#"{"name":"Watcher","nodes":[{"id":"a","type":"logic_block","data":{}}],"edges":[],"oaiyToolHook":{"tool":"run_command","mode":"before","flowId":"hook"}}"#);
    put(&src.0, "flows/tool.json", br#"{"name":"Lookup","nodes":[],"edges":[],"oaiyTool":{"name":"lookup_caller","description":"d","flowId":"tool"}}"#);
    let out = TempDir::new("describe-out");
    let file = out.0.join("d.oaiybackup");
    make(&src.0, &file);
    let dst = TempDir::new("describe-dst");
    let preview = restore::inspect(&dst.0, &file, PASS, &options()).unwrap();
    let what = |name: &str| preview.items.iter().find(|i| i.name == name).unwrap_or_else(|| panic!("{name} is listed")).what.clone();
    let template = what("templates/rig.json");
    for part in ["rig.exe --serve", "install script install-rig.ps1", "writes 1 script file(s)", "deletes 1 path(s)", "STARTS with OAIY", "sets 2 environment variable(s)", "LD_PRELOAD", "runs in C:/work"] {
        assert!(template.contains(part), "{part} is said of the template: {template}");
    }
    assert!(!template.contains("x.so"), "an environment variable's value is not shown, only its name");
    assert!(what("flows/hook.json").contains("runs before the Agent's \"run_command\" tool"), "{}", what("flows/hook.json"));
    assert!(what("flows/tool.json").contains("offered to the Agent as the tool \"lookup_caller\""), "{}", what("flows/tool.json"));
}

#[test]
fn decrypting_stops_at_the_deadline_by_itself_and_does_not_wait_for_the_checks_that_follow() {
    let src = TempDir::new("deadline-src");
    realistic(&src.0, "A");
    let out = TempDir::new("deadline-out");
    let file = out.0.join("d.oaiybackup");
    make(&src.0, &file);
    let scratch = TempDir::new("deadline-scratch");
    // A deadline that has already passed stops the decryption itself, before any of the ZIP is looked at.
    let expired = Budget::within(std::time::Duration::ZERO);
    let err = container::decrypt_to_file(&file, PASS, &scratch.0.join("late.zip"), 1 << 30, &expired).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Timeout, "{err}");
    // With time, the same file decrypts.
    let (_, len) = container::decrypt_to_file(&file, PASS, &scratch.0.join("in-time.zip"), 1 << 30, &Budget::unlimited()).unwrap();
    assert!(len > 0);
}

// ---- the backup and the updater live side by side --------------------------------------------------------------

/// The paths of the routes a router file adds (not its tests' stand-ins): what follows each `.route("`.
fn route_paths(source: &str) -> Vec<String> {
    source.split("#[cfg(test)]").next().unwrap().split(".route(\"").skip(1).map(|rest| rest.split('"').next().unwrap().to_string()).collect()
}

#[tokio::test]
async fn the_backups_routes_and_the_updaters_merge_without_overlapping_and_both_answer() {
    use tower::ServiceExt as _;
    let dir = TempDir::new("side-by-side");
    let updater = crate::update::Updater::new("0.1.0", crate::update::FeedSource::production(), std::time::Instant::now());
    // Merged in either order (axum refuses two routes for one path and method), each answers where it says.
    for first_the_updater in [true, false] {
        let app = if first_the_updater {
            axum::Router::new().merge(crate::update::routes::router(updater.clone())).merge(routes::router(dir.0.clone()))
        } else {
            axum::Router::new().merge(routes::router(dir.0.clone())).merge(crate::update::routes::router(updater.clone()))
        };
        for (path, key) in [("/api/update/status", "currentVersion"), ("/api/backup/status", "lastBackupOk")] {
            let response = app.clone().oneshot(axum::http::Request::builder().uri(path).body(axum::body::Body::empty()).unwrap()).await.unwrap();
            assert_eq!(response.status().as_u16(), 200, "{path}");
            let body: serde_json::Value = serde_json::from_slice(&axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
            assert!(body.get(key).is_some(), "{path} answers with its own status: {body}");
        }
    }
}

#[test]
fn no_route_belongs_to_both_the_backup_and_the_updater() {
    let update = route_paths(&source_text(include_str!("../update/routes.rs")));
    let backup = route_paths(&source_text(include_str!("routes.rs")));
    assert_eq!(update.len(), 3, "{update:?}");
    assert_eq!(backup.len(), 8, "{backup:?}");
    for path in &update {
        assert!(path.starts_with("/api/update/") && !routes::is_backup_path(path), "{path}");
        assert!(!backup.contains(path), "{path} is in both");
    }
    for path in &backup {
        assert!(routes::is_backup_path(path) && !path.starts_with("/api/update/"), "{path}");
    }
}

#[test]
fn the_window_registers_each_command_plugin_and_item_of_both_features_once() {
    let lib = source_text(include_str!("../lib.rs"));
    // Commands: a name twice in the handler list is a command that is silently the second one.
    let list = lib[lib.find("generate_handler![").expect("the handler list")..].split("])").next().unwrap();
    let names: Vec<&str> = list.lines().skip(1).map(|l| l.trim().trim_end_matches(',')).filter(|l| !l.is_empty() && !l.starts_with("//")).collect();
    let mut seen = std::collections::BTreeSet::new();
    for name in &names {
        assert!(seen.insert(name.rsplit("::").next().unwrap()), "{name} is registered twice");
    }
    for want in ["backup_create", "backup_restore_inspect", "backup_restore_stage", "backup_undo_stage", "backup_discard_pending", "backup_restart_to_apply", "update_check", "update_download", "update_install", "set_update_auto_check"] {
        assert!(seen.contains(want), "{want} is registered");
    }
    // Plugins, schemes and managed state: each once.
    let unique = |what: &str, lines: Vec<&str>| {
        let mut seen = std::collections::BTreeSet::new();
        for line in &lines {
            assert!(seen.insert(*line), "{what} {line:?} is there twice");
        }
        lines.len()
    };
    assert!(unique("plugin", lib.lines().map(str::trim).filter(|l| l.starts_with(".plugin(")).collect()) >= 5, "the plugins are there");
    assert!(unique("scheme", lib.lines().map(str::trim).filter(|l| l.starts_with(".register_uri_scheme_protocol(")).collect()) >= 2);
    unique("managed state", lib.lines().map(str::trim).filter(|l| l.contains(".manage(")).collect());
    // The window's permissions: the dialogs the backup opens are granted once, and nothing of the updater's plugin is granted to a webview.
    let capabilities: serde_json::Value = serde_json::from_str(include_str!("../../capabilities/default.json")).unwrap();
    let permissions: Vec<&str> = capabilities["permissions"].as_array().unwrap().iter().map(|p| p.as_str().unwrap()).collect();
    assert_eq!(unique("permission", permissions.clone()), permissions.len());
    assert_eq!(permissions.iter().filter(|p| **p == "dialog:default").count(), 1);
    assert!(permissions.iter().all(|p| !p.starts_with("updater:")), "the updater plugin's own commands answer nobody: {permissions:?}");
    assert_eq!(capabilities["windows"], serde_json::json!(["main"]));
    // The tray: each item once.
    let tray = source_text(include_str!("../tray.rs"));
    let ids: Vec<&str> = tray.split("MenuItem::with_id(handle, \"").skip(1).map(|rest| rest.split('"').next().unwrap()).collect();
    assert_eq!(ids, ["open", "update-check", "quit"], "the tray's items");
    // The Agent's page is found one way, for the updater's ask to save its work and for the backup's ask for its storage.
    let commands = source_text(include_str!("commands.rs"));
    let update = source_text(include_str!("../update/gui.rs"));
    assert!(commands.contains("crate::embed::agent_webview(&self.app)") && update.contains("crate::embed::agent_webview(app)"));
    assert!(!commands.contains("get_webview("), "the backup does not look the Agent's page up by its label itself");
    // Both features' routes are merged once each into the local API.
    let http = source_text(include_str!("../http.rs"));
    assert_eq!(http.matches(".merge(crate::update::routes::router(updater))").count(), 1);
    assert_eq!(http.matches(".merge(crate::backup::routes::router(backup_data_dir))").count(), 1);
}

#[test]
fn a_backup_asks_the_activity_the_updater_was_given_and_the_two_agree() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let updater = crate::update::Updater::new("0.1.0", crate::update::FeedSource::production(), std::time::Instant::now());
    // Nothing has said what the app is doing: still starting, for a backup as for an update.
    assert!(updater.activity().is_none());
    assert_eq!(Busy::look(updater.activity().as_deref()).codes(), ["starting"]);
    // The desktop gives the updater its probes once; a backup sees whatever they see.
    let probes = std::sync::Arc::new(Fake::default());
    updater.set_activity(probes.clone());
    assert!(!Busy::look(updater.activity().as_deref()).is_busy());
    probes.set(|s| s.tasks = 1);
    let later = std::time::Instant::now() + std::time::Duration::from_secs(3600);
    assert_eq!(Busy::look(updater.activity().as_deref()).codes(), ["agentTask"]);
    assert_eq!(updater.blockers_fresh(later).iter().map(|b| b.code).collect::<Vec<_>>(), ["agentTask"], "and the updater says the same");
    probes.set(|s| {
        s.tasks = 0;
        s.phone = Some(LineState::Live { plugin: "Aokie Phone Bridge".into(), count: 1 });
    });
    assert_eq!(Busy::look(updater.activity().as_deref()).codes(), ["phoneCall"]);
    assert_eq!(updater.blockers_fresh(later).iter().map(|b| b.code).collect::<Vec<_>>(), ["phoneCall"]);
}
