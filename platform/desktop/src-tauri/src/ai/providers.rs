//! AI provider store — configured chat providers whose API keys live on this
//! machine and are NEVER handed back over the wire.
//!
//! This mirrors [`crate::bridge::pairing`]'s secret-store pattern: a FULL record
//! that serializes the secret to disk ([`AiProvider`], with `api_key`) and a
//! secret-free PUBLIC projection ([`AiProviderPublic`], with only `has_key`) that
//! every API response uses. OAIY has no OS keyring, so the key is a plaintext
//! field in `providers.json` — the same trust model as the plaintext `hf-token`
//! file; it is the full/public split, not encryption, that keeps the key out of
//! responses. The file itself is written owner-only ([`crate::secret_file`]).
//!
//! Deliberately flat vs. the FormLogic reference: no reusable-secret / alias
//! indirection (a provider holds its own key), no OS keyring, no Codex virtual
//! providers, no `custom` protocol. Only `openai` and `anthropic`.

use crate::secret_file::{keep_aside, kept_aside, read_text_patiently, Patience, Retry, Text};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The wire dialect the chat proxy speaks to a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// OpenAI Chat Completions shape (also LM Studio / llama.cpp / Ollama `/v1`).
    #[default]
    OpenAi,
    /// Anthropic Messages — the proxy normalizes req/resp to/from the OpenAI shape.
    Anthropic,
}

/// A capability a provider can serve. Empty on a provider == "all" (legacy/OpenAI
/// convenience). All variants round-trip so hand-authored profiles don't get
/// dropped on load, but only `Chat` has a proxy path in v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Chat,
    Transcription,
    Speech,
    Embeddings,
    Realtime,
}

fn default_true() -> bool {
    true
}

/// FULL provider record — persisted, INCLUDING the plaintext `api_key`. Never
/// hand this to an HTTP response; use [`AiProvider::public`]. (Mirrors
/// [`crate::bridge::pairing`]'s `PairedToken`.)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiProvider {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default)]
    pub protocol: Protocol,
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Empty == supports all capabilities.
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// The user marked this an on-device endpoint → egress may allow a loopback /
    /// private / `http` target (an ollama / LM Studio server). Off by default.
    #[serde(default)]
    pub allow_local: bool,
    /// PLAINTEXT credential. No OS keyring in OAIY (hf-token precedent). Protected
    /// ONLY by the full/public split below — it never serializes into a response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl AiProvider {
    pub fn supports(&self, cap: Capability) -> bool {
        self.capabilities.is_empty() || self.capabilities.contains(&cap)
    }

    pub fn has_key(&self) -> bool {
        self.api_key.as_deref().is_some_and(|k| !k.trim().is_empty())
    }

    /// Secret-free projection. `api_key` becomes `has_key: bool`.
    pub fn public(&self) -> AiProviderPublic {
        AiProviderPublic {
            id: self.id.clone(),
            name: self.name.clone(),
            category: self.category.clone(),
            protocol: self.protocol,
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            capabilities: self.capabilities.clone(),
            enabled: self.enabled,
            allow_local: self.allow_local,
            has_key: self.has_key(),
        }
    }
}

/// Secret-free view for EVERY API response. Drops `api_key`, exposes `hasKey`.
/// Serialize-only — there is no way to represent the secret in this type.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiProviderPublic {
    pub id: String,
    pub name: String,
    pub category: Option<String>,
    pub protocol: Protocol,
    pub base_url: String,
    pub model: Option<String>,
    pub capabilities: Vec<Capability>,
    pub enabled: bool,
    pub allow_local: bool,
    /// Presence only — never the key.
    pub has_key: bool,
}

/// Upsert wire body. The client never holds the key, so CRUD omits it: `api_key`
/// is set ONLY via [`ProviderStore::set_key`], and an edit PRESERVES the existing
/// key (so a UI can save non-secret fields without resending the secret).
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiProviderInput {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub protocol: Protocol,
    pub base_url: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub allow_local: bool,
}

/// On-disk shape (versioned + explicit, mirrors pairing's `PersistShape`).
#[derive(Serialize, Deserialize, Default)]
struct PersistShape {
    #[serde(default)]
    providers: Vec<AiProvider>,
}

pub struct ProviderStore {
    providers: Vec<AiProvider>,
    /// None == in-memory (persist is a no-op), mirroring `PairingManager`.
    path: Option<PathBuf>,
    /// What went wrong with the file, in words for the person: said where the providers are listed.
    problem: Option<String>,
    /// The file is still at `path` and was not read, so saving would write over keys nobody has seen.
    keep_out: bool,
    patience: Patience,
    /// Set while saving is kept out: when the file is read again.
    retry: Option<Retry>,
}

pub type ProviderStoreHandle = Arc<Mutex<ProviderStore>>;

pub fn new_handle() -> ProviderStoreHandle {
    Arc::new(Mutex::new(ProviderStore::new()))
}

pub fn open_handle(path: PathBuf) -> ProviderStoreHandle {
    handle_of(ProviderStore::open(path))
}

/// `store` to share, and a file it could not read is read again until it can be. On a thread of its own, since a flow
/// that needs a provider should not wait for someone to open the page, and the gateway's reads take no `&mut`.
fn handle_of(store: ProviderStore) -> ProviderStoreHandle {
    let unread = store.retry.is_some();
    let handle = Arc::new(Mutex::new(store));
    if unread {
        let store = Arc::downgrade(&handle);
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(250));
            let Some(store) = store.upgrade() else { return };
            let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
            store.read_again();
            if store.retry.is_none() {
                return;
            }
        });
    }
    handle
}

/// Max plaintext key length — a header-safety bound.
const MAX_KEY_BYTES: usize = 8 * 1024;
const MAX_BASE_URL_BYTES: usize = 4096;

impl ProviderStore {
    pub fn new() -> Self {
        Self { providers: Vec::new(), path: None, problem: None, keep_out: false, patience: Patience::default(), retry: None }
    }

    /// Load + harden from `path`. Persisted data is untrusted (old / hand-edited
    /// files bypass `upsert`'s validator): drop invalid ids and de-dup by id.
    ///
    /// This store holds API keys, so only an absent file is an empty store. A
    /// file that EXISTS but fails to parse (one byte that is not UTF-8 does) is
    /// preserved, as `providers.json.corrupt` (`.corrupt.1`, and so on: no earlier
    /// one is replaced), before we start empty: discarding a hand-repairable file
    /// and overwriting it on the next `persist()` would destroy the user's
    /// credentials. One that cannot be read at all (a backup tool holding it, say)
    /// may be perfectly good, so it stays where it is, nothing is saved over it,
    /// and it is read again until it can be. Either way the person is told, in
    /// the log and where the providers are listed.
    pub fn open(path: PathBuf) -> Self {
        Self::open_patiently(path, Patience::default())
    }

    /// [`ProviderStore::open`], waiting for a file that is busy as `patience` says.
    pub fn open_patiently(path: PathBuf, patience: Patience) -> Self {
        let pauses = patience.start.clone();
        let mut store = Self { providers: Vec::new(), path: Some(path), problem: None, keep_out: false, patience, retry: None };
        store.load(&pauses);
        store
    }

    /// Read the file into the store, waiting as `pauses` say for one that is busy. What it holds takes the place of
    /// what is in memory: an edit made while it could not be read was refused, and the person was told.
    fn load(&mut self, pauses: &[Duration]) {
        let Some(path) = self.path.clone() else { return };
        let (raw, damage) = match read_text_patiently(&path, pauses) {
            Text::Missing => (Vec::new(), None),
            Text::Text(text) => match serde_json::from_str::<PersistShape>(&text) {
                Ok(doc) => (doc.providers, None),
                Err(e) => (Vec::new(), Some(e.to_string())),
            },
            Text::Undecodable(why) => (Vec::new(), Some(why)),
            Text::Unreadable(e) => {
                return self.kept_out(format!(
                    "The saved providers could not be read ({e}), so they are missing here: another program may have the file open, or its permissions may not let OAIY read it. Nothing will be saved over {}, and it is read again until it can be.",
                    path.display()
                ));
            }
        };
        let mut seen = HashSet::new();
        self.providers = raw
            .into_iter()
            .filter(|p| valid_id(&p.id))
            .filter(|p| seen.insert(p.id.clone()))
            .collect();
        match damage.map(|why| (keep_aside(&path), why)) {
            None => {
                self.problem = None;
                if self.keep_out {
                    log::info!("ai: {} could be read again", path.display());
                }
            }
            Some((Ok(aside), why)) => {
                let moved = format!(
                    "The saved providers could not be read ({why}), so they are missing here. Nothing is lost: the file is kept as {}, to repair by hand or restore from a backup.",
                    aside.display()
                );
                log::error!("ai: {moved}");
                self.problem = Some(moved);
            }
            Some((Err(moved), why)) => {
                return self.kept_out(format!(
                    "The saved providers could not be read ({why}), and {} could not be moved aside ({moved}). Nothing will be saved over it: repair it, or move it aside yourself, and it is read again until that is done.",
                    path.display()
                ));
            }
        }
        self.keep_out = false;
        self.retry = None;
    }

    /// The file is still there and nothing is known of it: `said` is why, and it is read again as `patience` says.
    fn kept_out(&mut self, said: String) {
        match &mut self.retry {
            Some(retry) => retry.failed(),
            None => {
                log::error!("ai: {said}");
                self.retry = Some(Retry::began(&self.patience));
            }
        }
        self.keep_out = true;
        self.problem = Some(said);
    }

    /// The file could not be read at the last try: read it again if it is time to.
    fn read_again(&mut self) {
        if self.retry.as_ref().is_some_and(Retry::due) {
            self.load(&[]);
        }
    }

    /// Public snapshot of every provider — never the key.
    pub fn list(&self) -> Vec<AiProviderPublic> {
        self.providers.iter().map(AiProvider::public).collect()
    }

    /// What the page that lists the providers says of why there are none: a file that cannot be read for now, or one
    /// kept aside, for as long as it is there and nothing has taken its place. A restart does not end that: a desktop
    /// that came back up after the move has no reason of its own to give, and the page would be empty with nothing said.
    pub fn problem(&self) -> Option<String> {
        if self.keep_out {
            return self.problem.clone();
        }
        if !self.providers.is_empty() {
            return None;
        }
        let aside = kept_aside(self.path.as_ref()?).pop()?;
        Some(self.problem.clone().unwrap_or_else(|| {
            format!("The saved providers could not be read when OAIY last tried, so they are missing here. Nothing is lost: the file is kept as {}, to repair by hand or restore from a backup.", aside.display())
        }))
    }

    /// Create or replace a provider by id. Validates id + base_url; PRESERVES an
    /// existing api_key when replacing (the CRUD body carries no secret).
    pub fn upsert(&mut self, input: AiProviderInput) -> Result<String, String> {
        if !valid_id(&input.id) {
            return Err("provider id must be lowercase letters, digits or dash (1–64 chars)".into());
        }
        if input.name.trim().is_empty() {
            return Err("provider name is required".into());
        }
        if input.base_url.len() > MAX_BASE_URL_BYTES {
            return Err("provider base URL is too long".into());
        }
        // Cheap, offline-safe syntactic check; the full SSRF/resolve check runs
        // per-request in the gateway's egress guard.
        crate::ai::egress::check_base_url_syntax(&input.base_url, input.allow_local)?;

        let existing_key = self.find(&input.id).and_then(|p| p.api_key.clone());
        let provider = AiProvider {
            id: input.id,
            name: input.name,
            category: input.category,
            protocol: input.protocol,
            base_url: input.base_url.trim_end_matches('/').to_string(),
            model: input.model,
            capabilities: input.capabilities,
            enabled: input.enabled,
            allow_local: input.allow_local,
            api_key: existing_key,
        };
        let id = provider.id.clone();
        match self.providers.iter_mut().find(|p| p.id == id) {
            Some(slot) => *slot = provider,
            None => self.providers.push(provider),
        }
        self.persist()?;
        Ok(id)
    }

    /// Remove a provider by id. Returns Err on an unknown id so the caller can
    /// 404, or when the removal could not be written to disk.
    pub fn delete(&mut self, id: &str) -> Result<(), String> {
        let before = self.providers.len();
        self.providers.retain(|p| p.id != id);
        if self.providers.len() == before {
            return Err(format!("unknown provider {id:?}"));
        }
        self.persist()
    }

    /// Set (or clear, on empty/None) a provider's plaintext key. Err on an
    /// unknown id, a rejected key, or a key that could not be written to disk —
    /// the caller must not report a key it will not still have after a restart.
    pub fn set_key(&mut self, id: &str, key: Option<&str>) -> Result<(), String> {
        let trimmed = key.map(str::trim).filter(|k| !k.is_empty());
        if let Some(k) = trimmed {
            if k.len() > MAX_KEY_BYTES || k.chars().any(char::is_control) {
                return Err("api key exceeds the safe limit or contains control characters".into());
            }
        }
        let provider = self
            .providers
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| format!("unknown provider {id:?}"))?;
        provider.api_key = trimmed.map(str::to_string);
        self.persist()
    }

    /// INTERNAL: the full record incl. `api_key`, for the gateway's outbound call
    /// only. Not exposed over HTTP.
    pub fn get_full(&self, id: &str) -> Option<AiProvider> {
        self.find(id).cloned()
    }

    /// The default provider for a capability: the first enabled, capable provider
    /// that holds a key or is an explicitly allowed local OpenAI-compatible endpoint. Used when a caller hits `/api/ai/v1/*` with no
    /// explicit id. Flat — no aliases.
    pub fn default_for(&self, cap: Capability) -> Option<AiProvider> {
        self.providers
            .iter()
            .find(|p| p.enabled && p.supports(cap) && (p.has_key() || (p.protocol == Protocol::OpenAi && p.allow_local)))
            .cloned()
    }

    fn find(&self, id: &str) -> Option<&AiProvider> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// Atomic replace, owner-only from the first byte. No-op when in-memory;
    /// otherwise the io error reaches the caller.
    ///
    /// This was previously infallible by construction (`if write(..).is_ok() {
    /// let _ = rename(..) }`), which made a failed save indistinguishable from a
    /// successful one: `set_key` still answered 204, `list()` still reported
    /// `hasKey: true` off the in-memory vec, chat worked for the rest of the
    /// session — and the key was simply absent on the next launch, with nothing
    /// logged anywhere. A data dir that exists but is not writable, a full disk,
    /// or a rename an indexer/AV blocks is enough to hit it.
    ///
    /// The in-memory change is deliberately NOT rolled back on failure: this
    /// writes the WHOLE store every time, so the next persist that succeeds
    /// carries the missed edits with it and a transient failure heals itself.
    fn persist(&self) -> Result<(), String> {
        let Some(path) = &self.path else { return Ok(()) };
        if self.keep_out {
            return Err(self.problem.clone().unwrap_or_else(|| "providers.json could not be read".to_string()));
        }
        let fail = |e: &dyn std::fmt::Display| {
            format!("could not save providers.json ({e}) — the change applies to this session only and is lost on restart")
        };
        let shape = PersistShape { providers: self.providers.clone() };
        let body = serde_json::to_string_pretty(&shape).map_err(|e| fail(&e))?;
        // The file holds the PLAINTEXT keys, so it is created private and the
        // staging copy is removed on a failed rename, never left as a second
        // file nothing reads or cleans (see `secret_file::write`).
        crate::secret_file::write(path, body).map_err(|e| fail(&e))
    }
}

impl Default for ProviderStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Lowercase letters/digits/dash, 1–64 chars (verbatim from the FormLogic reference).
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str) -> AiProviderInput {
        AiProviderInput {
            id: id.into(),
            name: "Test".into(),
            category: None,
            protocol: Protocol::OpenAi,
            base_url: "https://api.openai.com".into(),
            model: Some("example-model".into()),
            capabilities: vec![],
            enabled: true,
            allow_local: false,
        }
    }

    #[test]
    fn upsert_then_list_never_leaks_the_key() {
        let mut s = ProviderStore::new();
        s.upsert(input("openai")).unwrap();
        s.set_key("openai", Some("sk-secret")).unwrap();
        // The public list carries has_key, never the secret.
        let list = s.list();
        assert_eq!(list.len(), 1);
        assert!(list[0].has_key);
        let json = serde_json::to_string(&list).unwrap();
        assert!(!json.contains("sk-secret"), "the key must never serialize into a response");
        assert!(!json.contains("apiKey"));
        // But the gateway can read it internally.
        assert_eq!(s.get_full("openai").unwrap().api_key.as_deref(), Some("sk-secret"));
    }

    #[test]
    fn editing_preserves_the_existing_key() {
        let mut s = ProviderStore::new();
        s.upsert(input("openai")).unwrap();
        s.set_key("openai", Some("sk-1")).unwrap();
        let mut edit = input("openai");
        edit.name = "Renamed".into();
        s.upsert(edit).unwrap(); // no key in the CRUD body
        assert_eq!(s.get_full("openai").unwrap().api_key.as_deref(), Some("sk-1"));
        assert_eq!(s.get_full("openai").unwrap().name, "Renamed");
    }

    #[test]
    fn default_for_requires_enabled_capable_and_keyed() {
        let mut s = ProviderStore::new();
        s.upsert(input("openai")).unwrap();
        assert!(s.default_for(Capability::Chat).is_none(), "no key yet");
        s.set_key("openai", Some("sk-1")).unwrap();
        assert_eq!(s.default_for(Capability::Chat).unwrap().id, "openai");
        // Disabling drops it from default selection.
        let mut off = input("openai");
        off.enabled = false;
        s.upsert(off).unwrap();
        assert!(s.default_for(Capability::Chat).is_none());
    }

    #[test]
    fn default_accepts_keyless_local_openai_but_not_disabled_or_anthropic() {
        let mut s = ProviderStore::new();
        let mut local = input("local-qwen");
        local.base_url = "http://127.0.0.1:8088".into();
        local.allow_local = true;
        s.upsert(local.clone()).unwrap();
        assert_eq!(s.default_for(Capability::Chat).unwrap().id, "local-qwen");
        local.enabled = false;
        s.upsert(local.clone()).unwrap();
        assert!(s.default_for(Capability::Chat).is_none());
        local.enabled = true;
        local.protocol = Protocol::Anthropic;
        s.upsert(local).unwrap();
        assert!(s.default_for(Capability::Chat).is_none());
    }

    #[test]
    fn invalid_ids_and_urls_are_refused() {
        let mut s = ProviderStore::new();
        let mut bad = input("Bad_Id");
        assert!(s.upsert(bad.clone()).is_err(), "uppercase/underscore id");
        bad = input("ok");
        bad.base_url = "ftp://x".into();
        assert!(s.upsert(bad.clone()).is_err(), "non-https base url");
        bad = input("ok");
        bad.base_url = "https://user:pw@api.example.com".into();
        assert!(s.upsert(bad).is_err(), "credentials embedded in url");
    }

    #[test]
    fn set_key_clears_on_empty_and_errors_on_unknown() {
        let mut s = ProviderStore::new();
        s.upsert(input("openai")).unwrap();
        s.set_key("openai", Some("sk-1")).unwrap();
        s.set_key("openai", Some("  ")).unwrap(); // whitespace clears
        assert!(!s.get_full("openai").unwrap().has_key());
        assert!(s.set_key("nope", Some("x")).is_err());
    }

    #[test]
    fn a_corrupt_store_is_preserved_not_silently_wiped() {
        let dir = std::env::temp_dir().join(format!("oaiy-ai-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("providers.json");
        std::fs::write(&path, b"{ this is not valid json").unwrap();
        let mut s = ProviderStore::open(path.clone());
        // Starts empty — but the unparseable bytes are moved aside, not destroyed,
        // so a user's hand-editable keys survive for repair.
        assert!(s.list().is_empty());
        assert!(path.with_extension("json.corrupt").exists(), "corrupt file preserved as .corrupt");
        // And a later write persists cleanly to the real path.
        s.upsert(input("openai")).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One byte a tool saved in Windows-1252 used to read as "no file": every provider gone with nothing said, and the
    /// next save wrote the store back without them, keys and all, with no copy.
    #[test]
    fn a_store_that_is_not_text_keeps_its_keys_and_says_so() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-bytes");
        let path = dir.0.join("providers.json");
        let mut body = serde_json::to_vec(&serde_json::json!({ "providers": [{
            "id": "openai", "name": "Caf_", "protocol": "openai", "baseUrl": "https://api.openai.com",
            "enabled": true, "capabilities": [], "apiKey": "sk-precious",
        }] })).unwrap();
        let at = body.iter().position(|b| *b == b'_').unwrap();
        body[at] = 0xE9;
        std::fs::write(&path, &body).unwrap();

        let mut s = ProviderStore::open(path.clone());
        assert!(s.list().is_empty());
        assert_eq!(std::fs::read(path.with_extension("json.corrupt")).unwrap(), body, "the keys are all there, aside");
        let said = s.problem().expect("the page that lists the providers says why they are missing");
        assert!(said.contains("providers.json.corrupt"), "{said}");
        // A save is a new file, beside the copy.
        s.upsert(input("anthropic")).unwrap();
        assert!(path.with_extension("json.corrupt").exists());
        assert!(ProviderStore::open(path).problem().is_none(), "a file that reads has nothing to say");
    }

    /// A read that fails for another reason (a backup tool holding the file at start) says nothing about the file.
    #[test]
    fn a_store_that_cannot_be_read_is_not_saved_over() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-held");
        let path = dir.0.join("providers.json");
        // (a folder where the file belongs: a read that fails as a held file's does, on every system)
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("still-here"), b"x").unwrap();
        let mut s = ProviderStore::open(path.clone());
        assert!(s.list().is_empty());
        assert!(s.problem().is_some_and(|p| p.contains("Nothing will be saved over")));
        let err = s.upsert(input("openai")).unwrap_err();
        assert!(err.contains("Nothing will be saved over"), "{err}");
        assert!(path.join("still-here").exists());
    }

    const SAVED: &str = r#"{ "providers": [{ "id": "openai", "name": "OpenAI", "protocol": "openai", "baseUrl": "https://api.openai.com", "enabled": true, "capabilities": [], "apiKey": "sk-precious" }] }"#;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// A file held at start (an antivirus scan, a backup) cost every provider for the session, and every save, until
    /// a restart. Read again by itself: the keys are back and saving is allowed, with no one having to do anything.
    #[test]
    fn a_file_that_could_not_be_read_is_read_again_and_saving_comes_back() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-later");
        let path = dir.0.join("providers.json");
        std::fs::create_dir(&path).unwrap();
        // (read again every 20 ms from the first failure)
        let mut s = ProviderStore::open_patiently(path.clone(), Patience { start: vec![], later: vec![ms(20)], window: Duration::ZERO });
        assert!(s.upsert(input("anthropic")).is_err(), "refused while it cannot be read");
        // Not time yet, or still held: nothing changes.
        s.read_again();
        std::thread::sleep(ms(40));
        s.read_again();
        assert!(s.problem().is_some_and(|p| p.contains("read again")), "{:?}", s.problem());
        assert!(s.upsert(input("anthropic")).is_err(), "still refused");

        std::fs::remove_dir_all(&path).unwrap();
        std::fs::write(&path, SAVED).unwrap();
        std::thread::sleep(ms(40));
        s.read_again();
        assert!(s.problem().is_none(), "{:?}", s.problem());
        assert_eq!(s.get_full("openai").unwrap().api_key.as_deref(), Some("sk-precious"));
        assert!(s.get_full("anthropic").is_none(), "what was refused is not kept: the file is what there is");
        s.upsert(input("groq")).unwrap();
        let again = ProviderStore::open(path);
        assert_eq!(again.get_full("openai").unwrap().api_key.as_deref(), Some("sk-precious"));
        assert!(again.get_full("groq").is_some());
    }

    /// Nobody has to open the page or start a flow for it to happen.
    #[test]
    fn the_handle_reads_a_file_again_by_itself() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-handle");
        let path = dir.0.join("providers.json");
        std::fs::create_dir(&path).unwrap();
        let handle = handle_of(ProviderStore::open_patiently(path.clone(), Patience { start: vec![], later: vec![ms(20)], window: Duration::ZERO }));
        std::fs::remove_dir_all(&path).unwrap();
        std::fs::write(&path, SAVED).unwrap();
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while handle.lock().unwrap().list().is_empty() {
            assert!(std::time::Instant::now() < until, "not read again: {:?}", handle.lock().unwrap().problem());
            std::thread::sleep(ms(20));
        }
    }

    /// A lock that lasts a moment costs nothing at all: it is waited for.
    #[test]
    fn a_file_held_for_a_moment_at_start_is_waited_for() {
        use crate::secret_file::testing::{make_unreadable, TempDir};
        let dir = TempDir::new("ai-providers-held-a-moment");
        let path = dir.0.join("providers.json");
        std::fs::write(&path, SAVED).unwrap();
        let Some(held) = make_unreadable(&path) else { return };
        let letting_go = std::thread::spawn(move || {
            std::thread::sleep(ms(60));
            drop(held);
        });
        let s = ProviderStore::open_patiently(path, Patience { start: vec![ms(40); 20], later: vec![Duration::from_secs(3_600)], window: Duration::ZERO });
        letting_go.join().unwrap();
        assert!(s.problem().is_none(), "{:?}", s.problem());
        assert_eq!(s.list().len(), 1);
    }

    /// PowerShell 5.1's `Out-File` writes UTF-16, and some editors put a mark on UTF-8. Neither is damage: they were
    /// moved aside and the keys went with them.
    #[test]
    fn a_file_in_another_encoding_is_read_and_stays_where_it_is() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-encodings");
        let path = dir.0.join("providers.json");
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain(SAVED.encode_utf16().flat_map(u16::to_le_bytes)).collect();
        for body in [[&[0xEF, 0xBB, 0xBF][..], SAVED.as_bytes()].concat(), utf16] {
            std::fs::write(&path, &body).unwrap();
            let s = ProviderStore::open(path.clone());
            assert_eq!(s.get_full("openai").unwrap().api_key.as_deref(), Some("sk-precious"));
            assert!(s.problem().is_none(), "{:?}", s.problem());
            assert!(path.exists() && kept_aside(&path).is_empty());
        }
    }

    /// A second damaged file used to replace the copy of the first, which held the keys of the first; backups leave
    /// `*.corrupt` out, so nothing else did. And the page said nothing after a restart.
    #[test]
    fn every_damaged_file_is_kept_and_the_notice_lasts_while_the_store_is_empty() {
        let dir = crate::secret_file::testing::TempDir::new("ai-providers-damaged-twice");
        let path = dir.0.join("providers.json");
        std::fs::write(&path, b"{ first").unwrap();
        let mut s = ProviderStore::open(path.clone());
        assert!(s.problem().is_some_and(|p| p.contains("providers.json.corrupt")));

        // A restart with nothing done: the same page, with the copy named.
        let s2 = ProviderStore::open(path.clone());
        let said = s2.problem().expect("an empty page says why");
        assert!(said.contains("providers.json.corrupt") && said.contains("Nothing is lost"), "{said}");

        // A provider added, and the page has something to show.
        s.upsert(input("anthropic")).unwrap();
        assert!(s.problem().is_none(), "{:?}", s.problem());

        // The same tool damages the new file.
        std::fs::write(&path, b"{ second").unwrap();
        let s3 = ProviderStore::open(path.clone());
        assert_eq!(std::fs::read(path.with_extension("json.corrupt")).unwrap(), b"{ first", "the first copy stands");
        assert_eq!(std::fs::read(path.with_extension("json.corrupt.1")).unwrap(), b"{ second");
        assert!(s3.problem().is_some_and(|p| p.contains("providers.json.corrupt.1")), "the newest is named: {:?}", s3.problem());

        // With the copies gone there is nothing to say.
        for aside in kept_aside(&path) {
            std::fs::remove_file(aside).unwrap();
        }
        assert!(s3.problem().is_none(), "{:?}", s3.problem());
    }

    #[test]
    fn persists_and_hardens_on_reopen() {
        let dir = std::env::temp_dir().join(format!("oaiy-ai-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("providers.json");
        {
            let mut s = ProviderStore::open(path.clone());
            s.upsert(input("openai")).unwrap();
            s.set_key("openai", Some("sk-persist")).unwrap();
        }
        let s2 = ProviderStore::open(path.clone());
        assert_eq!(s2.list().len(), 1);
        assert_eq!(s2.get_full("openai").unwrap().api_key.as_deref(), Some("sk-persist"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_save_is_reported_and_leaves_no_plaintext_tmp() {
        // A directory standing where providers.json belongs makes the rename fail
        // exactly as a read-only data dir or a locked target does in the field,
        // without needing real permissions in a test.
        let dir = std::env::temp_dir().join(format!("oaiy-ai-unwritable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("providers.json");
        std::fs::create_dir_all(&path).unwrap();
        let mut s = ProviderStore::open(path.clone());

        let err = s.upsert(input("openai")).unwrap_err();
        assert!(err.contains("providers.json"), "the io failure must reach the caller: {err}");
        // The 204-with-a-lost-key case: set_key must not answer Ok for a secret
        // that never reached the disk.
        assert!(s.set_key("openai", Some("sk-secret")).is_err(), "an unsaved key must not report success");
        // Not rolled back — the edit stays live for the session so a later
        // persist that succeeds still carries it to disk.
        assert!(s.get_full("openai").unwrap().has_key());
        assert!(s.delete("openai").is_err(), "an unsaved delete must not report success");
        let left: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(left, ["providers.json"], "the plaintext key must not survive in a staging file: {left:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_keys_are_written_owner_only_in_a_folder_that_is_ours_alone() {
        use crate::secret_file::testing::{assert_private, assert_private_dir, TempDir};
        let dir = TempDir::new("providers");
        let path = dir.0.join("ai").join("providers.json");
        let mut s = ProviderStore::open(path.clone());
        s.upsert(input("openai")).unwrap();
        s.set_key("openai", Some("sk-private")).unwrap();
        assert_private(&path);
        assert_private_dir(path.parent().unwrap());
        // Every later save replaces the file whole and keeps it private.
        s.set_key("openai", Some("sk-rotated")).unwrap();
        assert_private(&path);
        assert!(std::fs::read_to_string(&path).unwrap().contains("sk-rotated"));
    }
}
