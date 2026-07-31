//! Answering the sealed AI turns a provider's web app relays to this desktop.
//!
//! The point of the lane: someone chats on the provider's website, and the
//! answer comes from THIS machine's account — a ChatGPT subscription or a
//! configured provider whose key never leaves here. The provider's backend
//! carries only sealed bytes and routing metadata; it cannot read the turn.
//!
//! ```text
//! POST   {pubkeyPath}                     publish this desktop's sealing key
//! GET    {pendingPath}?instanceId=&wait=  long-poll for sealed turns
//! POST   {claimPath}                      take one, exactly-once
//! POST   {framesPath}                     append sealed frames
//! POST   {completePath}                   done | failed
//! ```
//!
//! Until the key is published the provider's web app has nothing to encrypt to
//! and refuses to send at all — which is the entire visible symptom of this
//! module being absent. The publish therefore runs on every poll cycle until it
//! is accepted, and is recorded so it does not repeat needlessly.
//!
//! Descriptor-driven like the rest of the connector layer: every path, the poll
//! timing and the batch size come from [`crate::link::descriptor`], so a second
//! provider with this lane is a JSON file. Nothing here names a product.
//!
//! One turn at a time, by construction: the claim is atomic server-side and
//! this loop processes strictly sequentially, so a long answer cannot overlap
//! another.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::ai::e2e::{E2eIdentity, E2eSessions};
use crate::ai::providers::Capability;
use crate::link::descriptor::{self, DesktopAiSpec};
use crate::link::{LinkHandle, LinkedAccount};

/// A typed refusal, reported to the requester as a sealed error frame.
///
/// The codes are the provider's own taxonomy: the web app renders each one as a
/// specific message and recovery action, so inventing new ones here would show
/// the user a blank failure.
#[derive(Debug, Clone)]
pub struct TunnelError {
    code: &'static str,
    message: String,
}

impl TunnelError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn code(&self) -> &'static str {
        self.code
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// How often a stream's accumulated text is sealed into a frame.
///
/// The model produces fragments far faster than a relay round trip, and one
/// POST per token would spend the whole turn on HTTP. Coalescing is invisible
/// to the reader — the browser appends whatever arrives — and still shows the
/// answer building rather than landing all at once.
const DELTA_FLUSH_INTERVAL: Duration = Duration::from_millis(220);
/// …or sooner once this much text is waiting.
const DELTA_FLUSH_BYTES: usize = 400;
/// Bound on the frames one turn may emit, so a runaway stream cannot hold the
/// lane open indefinitely.
const MAX_STREAM_FRAMES: u64 = 4096;

/// Everything one claimed turn needs to talk back to the provider.
///
/// Bundled because a streaming answer posts frames from several places, and
/// threading five parameters through each of them invites getting one wrong —
/// the instance id in particular, which every call must carry.
struct Lane<'a> {
    http: reqwest::Client,
    account: &'a LinkedAccount,
    spec: &'a DesktopAiSpec,
    instance: &'a str,
    id: &'a str,
}

impl Lane<'_> {
    async fn post_frame(&self, envelope: &str) -> Result<(), String> {
        let url = crate::link::oauth::join(
            &self.account.base_url,
            &self.spec.frames_path.replace("{id}", self.id),
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.account.credential)
            .json(&json!({ "instanceId": self.instance, "envelope": envelope }))
            .send()
            .await
            .map_err(|e| format!("could not post a frame: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("the lane refused a frame: HTTP {}", resp.status().as_u16()));
        }
        Ok(())
    }
}

/// What this desktop can answer a turn with.
pub struct AiSources {
    pub providers: crate::ai::providers::ProviderStoreHandle,
    pub codex: crate::ai::codex::CodexHandle,
}

/// One sealed turn waiting for this desktop.
///
/// The id field is `requestId`, NOT `id`, and the batch is `requests`, not
/// `commands`. Both differ from the command relay next door, and getting either
/// wrong fails silently: the batch does not deserialize, and every turn the
/// user asks for expires as "no desktop picked it up in time".
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AiRequest {
    #[serde(rename = "requestId")]
    id: String,
    /// `chat`, `models` or `providers`.
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    provider_id: Option<String>,
    /// The browser's per-request public key, in routing PLAINTEXT — the shared
    /// key cannot be derived without it, so it cannot live inside the envelope.
    #[serde(default)]
    eph_pub: Option<String>,
    #[serde(default)]
    envelope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PendingReply {
    #[serde(default)]
    requests: Vec<AiRequest>,
}

/// The tunnel's own state: this desktop's sealing identity and the open
/// per-request sessions.
pub struct AiTunnel {
    identity: E2eIdentity,
    sessions: E2eSessions,
    sources: AiSources,
    /// Records the key the provider last accepted, so the publish is not
    /// repeated on every cycle once it has landed.
    published_marker: std::path::PathBuf,
    /// Log-once latch, so a provider that is down does not fill the log.
    publish_note: std::sync::Mutex<Option<String>>,
}

impl AiTunnel {
    pub fn new(data_dir: &std::path::Path, sources: AiSources) -> Result<Arc<Self>, String> {
        let identity = E2eIdentity::load_or_create(data_dir)?;
        Ok(Arc::new(Self {
            identity,
            sessions: E2eSessions::new(),
            sources,
            published_marker: data_dir.join("desktop-e2e-published.json"),
            publish_note: std::sync::Mutex::new(None),
        }))
    }

    pub fn public_key_b64(&self) -> String {
        self.identity.public_key_b64()
    }
}

/// Poll, claim, answer — for as long as a link with this lane exists.
///
/// Runs on its own thread with its own runtime rather than borrowing the
/// caller's: the AI gateway is async, the Codex agent is blocking, and this
/// worker outlives every request either of them serves.
pub fn spawn(store: LinkHandle, sources: AiSources) {
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                log::error!("the AI tunnel could not start a runtime: {e}");
                return;
            }
        };
        // The identity is minted on first use and must not be re-minted: every
        // browser that has chatted with this desktop pinned the public half.
        let tunnel = match AiTunnel::new(store.data_dir(), sources) {
            Ok(t) => t,
            Err(e) => {
                log::error!("the AI tunnel has no usable identity: {e}");
                return;
            }
        };
        runtime.block_on(async move {
            loop {
                let Some(account) = store.account() else {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                };
                let Some(spec) = descriptor::find(store.data_dir(), &account.connector_id)
                    .and_then(|d| d.desktop_ai)
                else {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    continue;
                };
                let instance = store.instance_id();

                tunnel
                    .publish_pubkey_if_needed(&account, &spec, &instance)
                    .await;
                match tunnel.poll_cycle(&account, &spec, &instance).await {
                    Ok(handled) => {
                        if handled == 0 {
                            tokio::time::sleep(Duration::from_millis(500)).await;
                        }
                    }
                    Err(e) => {
                        log::warn!("AI tunnel poll: {e}");
                        tokio::time::sleep(Duration::from_secs(spec.error_backoff_seconds)).await;
                    }
                }
            }
        });
    });
}

fn client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| format!("could not build the tunnel client: {e}"))
}

impl AiTunnel {
    /// Publish the sealing key when the provider does not already hold THIS one.
    ///
    /// Best effort and repeated: the provider needs a connection row first, so
    /// the first attempt after a link legitimately 404s, and giving up there
    /// would leave the website saying this desktop has published no key — which
    /// is exactly the failure this lane exists to fix.
    async fn publish_pubkey_if_needed(
        &self,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
    ) {
        let pubkey = self.identity.public_key_b64();
        if let Ok(text) = std::fs::read_to_string(&self.published_marker) {
            let already = serde_json::from_str::<Value>(&text)
                .ok()
                .is_some_and(|v| {
                    v.get("instanceId").and_then(Value::as_str) == Some(instance)
                        && v.get("publicKey").and_then(Value::as_str) == Some(pubkey.as_str())
                        && v.get("baseUrl").and_then(Value::as_str)
                            == Some(account.base_url.as_str())
                });
            if already {
                return;
            }
        }
        match self.publish(account, spec, instance, &pubkey).await {
            Ok(()) => {
                let body = json!({
                    "instanceId": instance,
                    "publicKey": pubkey,
                    "baseUrl": account.base_url,
                    "publishedAt": chrono::Utc::now().to_rfc3339(),
                });
                let tmp = self.published_marker.with_extension("tmp");
                if let Err(e) = std::fs::write(&tmp, body.to_string().as_bytes())
                    .and_then(|()| std::fs::rename(&tmp, &self.published_marker))
                {
                    log::warn!("AI tunnel could not record the published key: {e}");
                }
                self.note_publish(None);
            }
            Err(e) => self.note_publish(Some(e)),
        }
    }

    async fn publish(
        &self,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
        pubkey: &str,
    ) -> Result<(), String> {
        let http = client(Duration::from_secs(20))?;
        let resp = http
            .post(crate::link::oauth::join(&account.base_url, &spec.pubkey_path))
            .bearer_auth(&account.credential)
            .json(&json!({ "instanceId": instance, "publicKey": pubkey }))
            .send()
            .await
            .map_err(|e| format!("could not reach the AI lane: {e}"))?;
        if resp.status().is_success() {
            return Ok(());
        }
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        let message = body
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the provider refused the key");
        Err(format!("publishing this desktop's chat key: HTTP {status}: {message}"))
    }

    fn note_publish(&self, note: Option<String>) {
        let mut g = self.publish_note.lock().unwrap_or_else(|e| e.into_inner());
        if *g != note {
            match &note {
                Some(message) => log::warn!("AI tunnel: {message}"),
                None => log::info!("AI tunnel: chat key published"),
            }
            *g = note;
        }
    }

    /// One long-poll and the turns it carried. Returns how many were handled.
    async fn poll_cycle(
        &self,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
    ) -> Result<usize, String> {
        // `instanceId` is not optional: without it the provider only offers rows
        // addressed to NO desktop, and its web app always pins a target when one
        // resolves — so the lane would poll happily and stay empty forever.
        let url = format!(
            "{}?wait={}&limit={}&instanceId={}",
            crate::link::oauth::join(&account.base_url, &spec.pending_path),
            spec.wait_seconds * 1000,
            spec.batch_limit,
            urlencode(instance),
        );
        let http = client(Duration::from_secs(spec.wait_seconds + 15))?;
        let resp = http
            .get(&url)
            .bearer_auth(&account.credential)
            .send()
            .await
            .map_err(|e| format!("could not reach the AI lane: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body: Value = resp.json().await.unwrap_or(Value::Null);
            let message = body
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("the AI lane refused the poll");
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(format!(
                    "the provider no longer accepts this desktop's key ({message}) — link again"
                ));
            }
            return Err(format!("HTTP {}: {message}", status.as_u16()));
        }
        let reply: PendingReply = resp
            .json()
            .await
            .map_err(|e| format!("the AI lane returned an unreadable batch: {e}"))?;

        let mut handled = 0usize;
        for request in reply.requests {
            if let Err(e) = self.serve(account, spec, instance, &request).await {
                log::warn!("AI turn {} could not be served: {e}", request.id);
            }
            handled += 1;
        }
        Ok(handled)
    }

    /// Claim one turn, answer it, and always report a terminal status.
    async fn serve(
        &self,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
        request: &AiRequest,
    ) -> Result<(), String> {
        let http = client(Duration::from_secs(30))?;
        let claim_url = crate::link::oauth::join(
            &account.base_url,
            &spec.claim_path.replace("{id}", &request.id),
        );
        let claimed = http
            .post(&claim_url)
            .bearer_auth(&account.credential)
            .json(&json!({ "instanceId": instance }))
            .send()
            .await
            .map_err(|e| format!("could not claim: {e}"))?;
        if !claimed.status().is_success() {
            // 409 is the ordinary "someone else got there first".
            return if claimed.status().as_u16() == 409 {
                Ok(())
            } else {
                Err(format!("claim refused: HTTP {}", claimed.status().as_u16()))
            };
        }

        let lane = Lane {
            http: client(Duration::from_secs(30))?,
            account,
            spec,
            instance,
            id: &request.id,
        };
        let outcome = self.answer(&lane, request).await;
        let (status, frame) = match outcome {
            Ok(final_plaintext) => match self.sessions.seal_outbound(&request.id, &final_plaintext) {
                Ok(envelope) => ("done", Some(envelope)),
                Err(e) => ("failed", self.seal_error(request, e.code(), e.message())),
            },
            Err(e) => ("failed", self.seal_error(request, e.code(), e.message())),
        };

        // The answer rides the LAST frame and must land BEFORE the status flip:
        // completing purges the lane's frames, so anything posted after it would
        // never be readable and the user would see an empty reply.
        if let Some(envelope) = frame {
            if let Err(e) = lane.post_frame(&envelope).await {
                log::warn!("AI turn {} final frame: {e}", request.id);
            }
        }
        let complete_url = crate::link::oauth::join(
            &account.base_url,
            &spec.complete_path.replace("{id}", &request.id),
        );
        // Exactly `done` or `failed`. Any other word is a 400, the row stays
        // claimed, and the requester waits out the TTL and is told it expired.
        let done = http
            .post(&complete_url)
            .bearer_auth(&account.credential)
            .json(&json!({ "instanceId": instance, "status": status }))
            .send()
            .await
            .map_err(|e| format!("could not report the outcome: {e}"))?;
        self.sessions.drop_thread(&request.id);
        if !done.status().is_success() {
            return Err(format!(
                "the AI lane refused the outcome: HTTP {}",
                done.status().as_u16()
            ));
        }
        Ok(())
    }

    /// Seal a typed failure so the requester learns WHY.
    ///
    /// When the request envelope never opened there is no session to count
    /// from, so this seals detached at outbound counter 0 — the browser has not
    /// seen an outbound frame for this turn either, so the counters agree.
    fn seal_error(&self, request: &AiRequest, code: &str, message: &str) -> Option<String> {
        let frame = json!({
            "v": 1,
            "type": "error",
            "kind": "error",
            "code": code,
            "message": message,
        })
        .to_string();
        if let Ok(envelope) = self.sessions.seal_outbound(&request.id, frame.as_bytes()) {
            return Some(envelope);
        }
        let eph = request.eph_pub.as_deref()?;
        let eph: [u8; 32] = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            eph.trim(),
        )
        .ok()?
        .try_into()
        .ok()?;
        crate::ai::e2e::seal_detached_envelope(
            &self.identity.secret_key_bytes(),
            &eph,
            crate::ai::e2e::DIR_DESKTOP_TO_BROWSER,
            0,
            frame.as_bytes(),
        )
        .ok()
    }

    /// Open the sealed turn and produce the plaintext to seal back.
    async fn answer(&self, lane: &Lane<'_>, request: &AiRequest) -> Result<Vec<u8>, TunnelError> {
        let eph_pub = request
            .eph_pub
            .as_deref()
            .ok_or_else(|| TunnelError::new("invalid_request", "the turn carries no chat key"))?;
        let envelope = request
            .envelope
            .as_deref()
            .ok_or_else(|| TunnelError::new("invalid_request", "the turn carries no envelope"))?;
        let plaintext = self
            .sessions
            .open_inbound(&self.identity, &request.id, eph_pub, envelope)
            .map_err(|e| TunnelError::new(e.code(), e.message()))?;
        let body: Value = serde_json::from_slice(&plaintext)
            .map_err(|_| TunnelError::new("sealed_envelope_invalid", "the sealed body is not JSON"))?;
        if body.get("v").and_then(Value::as_u64) != Some(1) {
            return Err(TunnelError::new(
                "sealed_envelope_invalid",
                "the sealed body has no v:1 marker",
            ));
        }
        let model = body.get("model").and_then(Value::as_str).map(str::to_string);
        if let Some(m) = model.as_deref() {
            if m.is_empty() || m.len() > 256 || m.chars().any(char::is_control) {
                return Err(TunnelError::new(
                    "model_unavailable",
                    "the requested model name is malformed",
                ));
            }
        }

        let kind = request.kind.as_deref().unwrap_or("chat");
        // `providers` enumerates the whole gateway and is asked for under a
        // placeholder provider id, so it must be answered BEFORE any attempt to
        // resolve one — resolving the placeholder would empty the web app's
        // provider dropdown and leave nobody able to pick anything.
        if kind == "providers" {
            return Ok(self.list_providers().await);
        }
        let provider_id = request
            .provider_id
            .as_deref()
            .filter(|p| !p.is_empty())
            .ok_or_else(|| TunnelError::new("invalid_request", "the turn names no provider"))?;

        match kind {
            "models" => self.list_models(provider_id).await,
            "chat" => self.chat(lane, provider_id, model.as_deref(), &body).await,
            other => Err(TunnelError::new(
                "invalid_request",
                format!("this desktop does not serve the AI request kind {other:?}"),
            )),
        }
    }

    /// The chat-servable providers, in the row shape the web app's mapper reads.
    ///
    /// Ids go out BARE. The web app prefixes its own namespace when it stores a
    /// choice, so sending an already-prefixed id produces a reference that
    /// resolves to nothing here — and every later chat fails as an unknown
    /// provider, which reads as a misconfigured account rather than a wiring bug.
    async fn list_providers(&self) -> Vec<u8> {
        let mut rows: Vec<Value> = Vec::new();
        {
            let store = self
                .sources
                .providers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for view in store.list() {
                let capabilities: Vec<String> = view
                    .capabilities
                    .iter()
                    .map(|c| {
                        serde_json::to_value(c)
                            .ok()
                            .and_then(|v| v.as_str().map(str::to_string))
                            .unwrap_or_default()
                    })
                    .collect();
                let mut row = json!({
                    "id": view.id,
                    "label": view.name,
                    "capabilities": capabilities,
                    "enabled": view.enabled,
                });
                if let Some(model) = view.model.as_deref().filter(|m| !m.is_empty()) {
                    row["model"] = json!(model);
                }
                rows.push(row);
            }
        }
        // The managed ChatGPT connector, offered only when it is actually signed
        // in — listing it otherwise offers a guaranteed failure.
        let codex = self.sources.codex.clone();
        if tokio::task::spawn_blocking(move || codex.models())
            .await
            .is_ok_and(|r| r.is_ok())
        {
            rows.push(json!({
                "id": crate::ai::codex::CODEX_PROVIDER_ID,
                "label": "ChatGPT via Codex",
                "capabilities": ["chat"],
                "enabled": true,
            }));
        }
        // Both `type` and `kind`: the peer's own frames are inconsistent about
        // which one they set, and its reader tries them in that order.
        serde_json::to_vec(&json!({ "v": 1, "type": "providers", "kind": "providers", "providers": rows }))
            .unwrap_or_default()
    }

    async fn list_models(&self, provider_id: &str) -> Result<Vec<u8>, TunnelError> {
        let raw = if is_codex(provider_id) {
            let codex = self.sources.codex.clone();
            tokio::task::spawn_blocking(move || codex.models())
                .await
                .map_err(|e| TunnelError::new("upstream_error", e.to_string()))?
                .map_err(|e| TunnelError::new("provider_unavailable", e.message()))?
        } else {
            let provider = self.resolve(provider_id)?;
            crate::ai::gateway::models(&provider)
                .await
                .map_err(|e| TunnelError::new("provider_unavailable", e.message()))?
        };
        Ok(serde_json::to_vec(&json!({
            "v": 1,
            "type": "models",
            "kind": "models",
            "models": normalize_models(&raw),
        }))
        .unwrap_or_default())
    }

    /// Answer one chat turn from this machine's own account, streaming the
    /// answer as it is produced.
    ///
    /// The terminal frame is where this is easy to get wrong. The reader takes
    /// a final frame's text as the WHOLE answer, replacing everything the
    /// deltas built — so once anything has streamed, the final must carry no
    /// text at all. A final holding, say, the last chunk would silently reduce
    /// a long reply to one word.
    async fn chat(
        &self,
        lane: &Lane<'_>,
        provider_id: &str,
        model: Option<&str>,
        body: &Value,
    ) -> Result<Vec<u8>, TunnelError> {
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .filter(|m| !m.is_empty())
            .cloned()
            .ok_or_else(|| TunnelError::new("invalid_request", "the turn carries no messages"))?;
        let mut chat_body = json!({ "messages": messages });
        if let Some(m) = model {
            chat_body["model"] = json!(m);
        }

        if is_codex(provider_id) {
            // The live-call aliases pin a model and a reasoning effort for a
            // phone call and refuse caller audio; a queued chat turn is not that,
            // and offering them here would be offering guaranteed failures.
            if provider_id != crate::ai::codex::CODEX_PROVIDER_ID {
                return Err(TunnelError::new(
                    "provider_unavailable",
                    format!("provider {provider_id:?} is for live calls and cannot serve chat"),
                ));
            }
            return self.stream_codex(lane, chat_body).await;
        }

        let provider = self.resolve(provider_id)?;
        if crate::ai::gateway::streamable(&provider) {
            chat_body["stream"] = json!(true);
            return self.stream_registry(lane, &provider, chat_body).await;
        }
        let completion = crate::ai::gateway::chat(&provider, chat_body)
            .await
            .map_err(|e| TunnelError::new("upstream_error", e.message()))?;
        Ok(final_with_completion(completion))
    }

    /// Stream the managed ChatGPT account's answer.
    ///
    /// The turn runs on a blocking thread and hands fragments back through a
    /// channel; this side coalesces them into frames. The buffered completion
    /// still comes back, so a turn that produced no fragments at all — a
    /// runtime with no delta stream — closes with the whole answer instead of
    /// nothing.
    async fn stream_codex(
        &self,
        lane: &Lane<'_>,
        chat_body: Value,
    ) -> Result<Vec<u8>, TunnelError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let codex = self.sources.codex.clone();
        let turn = tokio::task::spawn_blocking(move || {
            codex.chat_streaming(&chat_body, None, |fragment| {
                // The receiver is dropped only when this task is abandoned;
                // a failed send just means nobody is streaming any more.
                let _ = tx.send(fragment.to_string());
            })
        });

        let mut streamed = 0u64;
        let mut pending = String::new();
        let mut since_flush = std::time::Instant::now();
        loop {
            match tokio::time::timeout(DELTA_FLUSH_INTERVAL, rx.recv()).await {
                Ok(Some(fragment)) => {
                    pending.push_str(&fragment);
                    if pending.len() < DELTA_FLUSH_BYTES && since_flush.elapsed() < DELTA_FLUSH_INTERVAL
                    {
                        continue;
                    }
                }
                // The turn ended and the sender dropped: flush the tail.
                Ok(None) => {
                    self.flush_delta(lane, &mut pending, &mut streamed).await?;
                    break;
                }
                Err(_) => {}
            }
            self.flush_delta(lane, &mut pending, &mut streamed).await?;
            since_flush = std::time::Instant::now();
        }

        let completion = turn
            .await
            .map_err(|e| TunnelError::new("upstream_error", e.to_string()))?
            .map_err(|e| TunnelError::new(codex_code(&e), e.message()))?;
        // Nothing streamed — a runtime that sends no fragments — so the whole
        // answer has to ride the terminal frame after all.
        Ok(if streamed == 0 {
            final_with_completion(completion)
        } else {
            final_after_stream()
        })
    }

    /// Stream a registry provider's answer, forwarding each upstream chunk.
    ///
    /// The chunk goes through VERBATIM, nested under `delta`. Spreading it at
    /// the frame's top level instead is silently dropped by the reader — no
    /// error, just an answer that never appears.
    async fn stream_registry(
        &self,
        lane: &Lane<'_>,
        provider: &crate::ai::providers::AiProvider,
        chat_body: Value,
    ) -> Result<Vec<u8>, TunnelError> {
        use futures_util::StreamExt;

        let upstream = crate::ai::gateway::chat_stream(provider, chat_body)
            .await
            .map_err(|e| TunnelError::new("upstream_error", e.message()))?;
        // A provider that ignores `stream: true` and answers with a plain JSON
        // body is not an error — treat it as the buffered case.
        let is_sse = upstream
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("text/event-stream"));
        if !is_sse {
            let body = upstream
                .text()
                .await
                .map_err(|e| TunnelError::new("upstream_error", e.to_string()))?;
            let completion: Value = serde_json::from_str(&body).map_err(|_| {
                TunnelError::new("upstream_error", "the provider returned neither a stream nor JSON")
            })?;
            return Ok(final_with_completion(completion));
        }

        let mut stream = upstream.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut sent = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|e| TunnelError::new("upstream_error", format!("stream read failed: {e}")))?;
            buf.extend_from_slice(&chunk);
            while let Some(end) = sse_event_end(&buf) {
                let event: Vec<u8> = buf.drain(..end).collect();
                for line in event.split(|b| *b == b'\n') {
                    let Some(data) = trim_ascii(line).strip_prefix(b"data:") else {
                        continue;
                    };
                    let data = trim_ascii(data);
                    if data == b"[DONE]" || data.is_empty() {
                        continue;
                    }
                    let Ok(delta) = serde_json::from_slice::<Value>(data) else {
                        continue;
                    };
                    sent += 1;
                    if sent > MAX_STREAM_FRAMES {
                        return Err(TunnelError::new(
                            "upstream_error",
                            "the provider's stream exceeded the frame cap",
                        ));
                    }
                    let frame = json!({ "v": 1, "type": "delta", "kind": "delta", "delta": delta });
                    let envelope = self
                        .sessions
                        .seal_outbound(lane.id, frame.to_string().as_bytes())
                        .map_err(|e| TunnelError::new(e.code(), e.message()))?;
                    // A dropped frame leaves a gap the reader can see in its
                    // own text; abandoning the turn over it would be worse.
                    if let Err(e) = lane.post_frame(&envelope).await {
                        log::warn!("AI turn {} delta {sent}: {e}", lane.id);
                    }
                }
            }
        }
        if sent == 0 {
            // Say so rather than closing a turn with no content and no reason:
            // an empty answer with a `done` status reads as the model having
            // nothing to say.
            return Err(TunnelError::new(
                "upstream_error",
                "the provider's stream carried no content",
            ));
        }
        Ok(final_after_stream())
    }

    /// Seal and post whatever text is waiting, if any.
    async fn flush_delta(
        &self,
        lane: &Lane<'_>,
        pending: &mut String,
        streamed: &mut u64,
    ) -> Result<(), TunnelError> {
        if pending.is_empty() {
            return Ok(());
        }
        if *streamed >= MAX_STREAM_FRAMES {
            pending.clear();
            return Ok(());
        }
        // The OpenAI streaming-chunk shape, because that is what the reader
        // digs into: `delta.choices[].delta.content`.
        let frame = json!({
            "v": 1,
            "type": "delta",
            "kind": "delta",
            "delta": { "choices": [{ "index": 0, "delta": { "content": pending.as_str() } }] },
        });
        let envelope = self
            .sessions
            .seal_outbound(lane.id, frame.to_string().as_bytes())
            .map_err(|e| TunnelError::new(e.code(), e.message()))?;
        pending.clear();
        *streamed += 1;
        if let Err(e) = lane.post_frame(&envelope).await {
            log::warn!("AI turn {} delta {streamed}: {e}", lane.id);
        }
        Ok(())
    }

    /// Fail closed: an unknown, disabled or keyless provider is refused by name
    /// rather than silently substituted for whatever else is configured.
    fn resolve(&self, provider_id: &str) -> Result<crate::ai::providers::AiProvider, TunnelError> {
        let provider = {
            let store = self
                .sources
                .providers
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            store.get_full(provider_id)
        };
        let provider = provider.ok_or_else(|| {
            TunnelError::new(
                "provider_unavailable",
                format!("this desktop has no provider named {provider_id:?}"),
            )
        })?;
        if !provider.supports(Capability::Chat) {
            return Err(TunnelError::new(
                "provider_unavailable",
                format!("provider {provider_id:?} is not configured for chat"),
            ));
        }
        if !provider.enabled {
            return Err(TunnelError::new(
                "provider_unavailable",
                format!("provider {provider_id:?} is turned off on this desktop"),
            ));
        }
        if !provider.has_key() && !provider.allow_local {
            return Err(TunnelError::new(
                "provider_unavailable",
                format!("provider {provider_id:?} has no API key on this desktop"),
            ));
        }
        Ok(provider)
    }
}

/// Every id the managed ChatGPT connector answers to, including the live-call
/// aliases — recognised here so they can be refused with a reason instead of
/// falling through to "no such provider".
fn is_codex(provider_id: &str) -> bool {
    provider_id == crate::ai::codex::CODEX_PROVIDER_ID
        || crate::ai::codex::LiveCallAlias::from_id(provider_id).is_some()
}

/// The terminal frame for a turn that streamed nothing: it carries the whole
/// answer, because nothing else did.
fn final_with_completion(completion: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "v": 1,
        "type": "final",
        "kind": "final",
        "completion": completion,
    }))
    .unwrap_or_default()
}

/// The terminal frame for a turn that already streamed its answer.
///
/// Deliberately EMPTY of text. The reader treats a final frame's text as the
/// whole answer and throws away everything the deltas built, so a completion
/// here would replace a long reply with whatever this frame happened to hold.
fn final_after_stream() -> Vec<u8> {
    br#"{"v":1,"type":"final","kind":"final"}"#.to_vec()
}

/// End of a complete SSE event (a blank line), if one is buffered.
fn sse_event_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2)
        .position(|w| w == b"\n\n")
        .map(|i| i + 2)
        .or_else(|| buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4))
}

fn trim_ascii(mut s: &[u8]) -> &[u8] {
    while matches!(s.first(), Some(b' ' | b'\t' | b'\r')) {
        s = &s[1..];
    }
    while matches!(s.last(), Some(b' ' | b'\t' | b'\r')) {
        s = &s[..s.len() - 1];
    }
    s
}

fn codex_code(e: &crate::ai::codex::CodexError) -> &'static str {
    match e {
        crate::ai::codex::CodexError::NotAuthenticated => "provider_unavailable",
        crate::ai::codex::CodexError::Unavailable(_) => "provider_unavailable",
        crate::ai::codex::CodexError::Rpc(_) => "upstream_error",
    }
}

/// Flatten a model catalogue into the plain `[{id, name?}]` array the web app
/// renders. It keeps object entries only, so an OpenAI `{data:[…]}` body passed
/// through verbatim leaves the dropdown empty with nothing explaining why.
fn normalize_models(value: &Value) -> Vec<Value> {
    let entries = value
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| value.get("models").and_then(Value::as_array))
        .or_else(|| value.as_array());
    let Some(entries) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())?;
            let mut out = json!({ "id": id });
            if let Some(name) = entry
                .get("displayName")
                .and_then(Value::as_str)
                .or_else(|| entry.get("display_name").and_then(Value::as_str))
                .or_else(|| entry.get("name").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
            {
                out["name"] = json!(name);
            }
            Some(out)
        })
        .collect()
}

fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pending_row_deserializes_from_the_providers_actual_wire_shape() {
        // The id is `requestId` and the batch is `requests` — both differ from
        // the command relay next door. Get either wrong and nothing fails
        // loudly: the batch will not parse, and every turn the user asks for
        // expires as "no desktop picked it up in time".
        let raw = json!({
            "requests": [{
                "requestId": "1f0c2a1e-0000-4000-8000-000000000001",
                "ownerUserId": "u1",
                "requestingUserId": "u1",
                "targetInstanceId": "oaiy-752737079f8640c29d59b38527f1fb87",
                "providerId": "openai-codex-agent",
                "kind": "chat",
                "ephPub": "eaYx7t4b+cmPEgMs3q3Q56B5OY/HhriMyEbsia+FpRo=",
                "envelope": "AAAA",
                "idempotencyKey": "ui-turn-1",
                "status": "pending",
                "claimedBy": null,
                "createdAt": "2026-08-01 09:15:00",
                "claimedAt": null,
                "finishedAt": null,
                "expiresAt": "2026-08-01 09:20:00"
            }]
        });
        let reply: PendingReply = serde_json::from_value(raw).expect("the real shape must parse");
        assert_eq!(reply.requests.len(), 1);
        let r = &reply.requests[0];
        assert_eq!(r.id, "1f0c2a1e-0000-4000-8000-000000000001");
        assert_eq!(r.kind.as_deref(), Some("chat"));
        assert_eq!(r.provider_id.as_deref(), Some("openai-codex-agent"));
        assert!(r.eph_pub.as_deref().unwrap().ends_with("pRo="));
    }

    #[test]
    fn an_empty_batch_is_the_ordinary_long_poll_timeout() {
        let empty: PendingReply = serde_json::from_value(json!({})).unwrap();
        assert!(empty.requests.is_empty());
        let none: PendingReply = serde_json::from_value(json!({ "requests": [] })).unwrap();
        assert!(none.requests.is_empty());
    }

    #[test]
    fn the_builtin_connector_describes_the_ai_lane_it_needs() {
        let d = descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let a = d.desktop_ai.expect("the connector must declare the AI lane");
        for path in [&a.claim_path, &a.frames_path, &a.complete_path] {
            assert!(path.contains("{id}"), "{path}");
        }
        // The publish and the poll are not per-turn; a placeholder there would
        // be sent to the provider literally.
        assert!(!a.pubkey_path.contains("{id}"));
        assert!(!a.pending_path.contains("{id}"));
        assert!(a.wait_seconds >= 5 && a.wait_seconds <= 60, "{}", a.wait_seconds);
    }

    #[test]
    fn every_live_call_alias_is_recognised_so_it_can_be_refused_with_a_reason() {
        // They are the same managed account, so falling through to "no such
        // provider" would be a lie; they are pinned for a phone call and cannot
        // serve a queued turn.
        assert!(is_codex(crate::ai::codex::CODEX_PROVIDER_ID));
        for alias in crate::ai::codex::LiveCallAlias::all() {
            assert!(is_codex(alias.id()), "{}", alias.id());
        }
        assert!(!is_codex("openai"));
        assert!(!is_codex(""));
    }

    #[test]
    fn a_model_catalogue_flattens_to_what_the_web_app_renders() {
        // It keeps object entries with an id only. Passing an OpenAI body
        // through verbatim leaves the dropdown empty with nothing saying why.
        let openai = json!({ "object": "list", "data": [{ "id": "gpt-5.5" }, { "id": "" }] });
        assert_eq!(normalize_models(&openai), vec![json!({ "id": "gpt-5.5" })]);

        let codex = json!({ "models": [{ "id": "gpt-5.6-luna", "displayName": "Luna" }] });
        assert_eq!(
            normalize_models(&codex),
            vec![json!({ "id": "gpt-5.6-luna", "name": "Luna" })]
        );
        // A shape with no entries at all is an empty list, not a panic.
        assert!(normalize_models(&json!({ "oops": true })).is_empty());
    }

    #[test]
    fn the_terminal_frame_after_a_stream_carries_no_text() {
        // The reader takes a final frame's text as the WHOLE answer and throws
        // away everything the deltas built. So a completion here would not
        // duplicate the reply — it would REPLACE it, and a final holding the
        // last chunk would cut a long answer down to one word.
        let after: Value = serde_json::from_slice(&final_after_stream()).unwrap();
        assert_eq!(after["kind"], "final");
        assert_eq!(after["type"], "final");
        assert!(after.get("completion").is_none(), "{after}");
        assert!(after.get("text").is_none(), "{after}");

        // …whereas a turn that streamed nothing must carry the whole answer,
        // or the reader is left with an empty reply and a done status.
        let buffered: Value =
            serde_json::from_slice(&final_with_completion(json!({ "choices": [] }))).unwrap();
        assert!(buffered.get("completion").is_some());
    }

    #[test]
    fn a_delta_frame_nests_the_chunk_where_the_reader_digs_for_it() {
        // The reader looks at `delta.choices[].delta.content`. A chunk spread
        // at the frame's top level is dropped in silence — no error, just an
        // answer that never appears.
        let frame = json!({
            "v": 1,
            "type": "delta",
            "kind": "delta",
            "delta": { "choices": [{ "index": 0, "delta": { "content": "Hi" } }] },
        });
        assert_eq!(frame["delta"]["choices"][0]["delta"]["content"], "Hi");
        assert!(frame.get("choices").is_none(), "the chunk must not be top-level");
        assert_eq!(frame["kind"], "delta");
    }

    #[test]
    fn sse_events_are_split_on_a_blank_line_in_either_newline_style() {
        assert_eq!(sse_event_end(b"data: {}\n\nrest"), Some(10));
        assert_eq!(sse_event_end(b"data: {}\r\n\r\nrest"), Some(12));
        // A partial event is not an event yet — draining it would lose bytes.
        assert_eq!(sse_event_end(b"data: {\"a\":1}"), None);
        assert_eq!(trim_ascii(b"  data: x \r"), b"data: x");
    }

    #[test]
    fn an_instance_id_is_safe_in_a_query_string() {
        assert_eq!(urlencode("oaiy-abc123"), "oaiy-abc123");
        assert_eq!(urlencode("a b&c=d"), "a%20b%26c%3Dd");
    }
}
