//! AI provider store — configured chat providers whose API keys live on this
//! machine and are NEVER handed back over the wire.
//!
//! This mirrors [`crate::bridge::pairing`]'s secret-store pattern: a FULL record
//! that serializes the secret to disk ([`AiProvider`], with `api_key`) and a
//! secret-free PUBLIC projection ([`AiProviderPublic`], with only `has_key`) that
//! every API response uses. OAIY has no OS keyring, so the key is a plaintext
//! field in `providers.json` — the same trust model as the plaintext `hf-token`
//! file; it is the full/public split, not encryption, that keeps the key out of
//! responses.
//!
//! Deliberately flat vs. the FormLogic reference: no reusable-secret / alias
//! indirection (a provider holds its own key), no OS keyring, no Codex virtual
//! providers, no `custom` protocol. Only `openai` and `anthropic`.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
}

pub type ProviderStoreHandle = Arc<Mutex<ProviderStore>>;

pub fn new_handle() -> ProviderStoreHandle {
    Arc::new(Mutex::new(ProviderStore::new()))
}

pub fn open_handle(path: PathBuf) -> ProviderStoreHandle {
    Arc::new(Mutex::new(ProviderStore::open(path)))
}

/// Max plaintext key length — a header-safety bound.
const MAX_KEY_BYTES: usize = 8 * 1024;
const MAX_BASE_URL_BYTES: usize = 4096;

impl ProviderStore {
    pub fn new() -> Self {
        Self { providers: Vec::new(), path: None }
    }

    /// Load + harden from `path`. Persisted data is untrusted (old / hand-edited
    /// files bypass `upsert`'s validator): drop invalid ids and de-dup by id.
    ///
    /// A file that EXISTS but fails to parse is preserved (renamed to
    /// `providers.json.corrupt`) before we start empty — this store holds API
    /// keys, so silently discarding a hand-repairable file and then overwriting
    /// it on the next `persist()` would destroy the user's credentials. An absent
    /// or unreadable file simply starts empty.
    pub fn open(path: PathBuf) -> Self {
        let raw: Vec<AiProvider> = match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str::<PersistShape>(&contents) {
                Ok(doc) => doc.providers,
                Err(_) => {
                    let backup = path.with_extension("json.corrupt");
                    let _ = std::fs::rename(&path, &backup);
                    Vec::new()
                }
            },
            Err(_) => Vec::new(), // absent or unreadable → empty
        };
        let mut seen = HashSet::new();
        let providers = raw
            .into_iter()
            .filter(|p| valid_id(&p.id))
            .filter(|p| seen.insert(p.id.clone()))
            .collect();
        Self { providers, path: Some(path) }
    }

    /// Public snapshot of every provider — never the key.
    pub fn list(&self) -> Vec<AiProviderPublic> {
        self.providers.iter().map(AiProvider::public).collect()
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
        self.persist();
        Ok(id)
    }

    /// Remove a provider by id. Returns Err on an unknown id so the caller can 404.
    pub fn delete(&mut self, id: &str) -> Result<(), String> {
        let before = self.providers.len();
        self.providers.retain(|p| p.id != id);
        if self.providers.len() == before {
            return Err(format!("unknown provider {id:?}"));
        }
        self.persist();
        Ok(())
    }

    /// Set (or clear, on empty/None) a provider's plaintext key.
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
        self.persist();
        Ok(())
    }

    /// INTERNAL: the full record incl. `api_key`, for the gateway's outbound call
    /// only. Not exposed over HTTP.
    pub fn get_full(&self, id: &str) -> Option<AiProvider> {
        self.find(id).cloned()
    }

    /// The default provider for a capability: the first enabled, capable provider
    /// that actually holds a key. Used when a caller hits `/api/ai/v1/*` with no
    /// explicit id. Flat — no aliases.
    pub fn default_for(&self, cap: Capability) -> Option<AiProvider> {
        self.providers
            .iter()
            .find(|p| p.enabled && p.supports(cap) && p.has_key())
            .cloned()
    }

    fn find(&self, id: &str) -> Option<&AiProvider> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// Atomic tmp + rename (mirrors `PairingManager::persist`). No-op when in-memory.
    fn persist(&self) {
        let Some(path) = &self.path else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let shape = PersistShape { providers: self.providers.clone() };
        if let Ok(body) = serde_json::to_string_pretty(&shape) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &body).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
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
            model: Some("gpt-4o-mini".into()),
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
}
