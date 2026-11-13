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
//! One turn of each kind at a time (contract §0.11): the loop claims every row,
//! in the batch's order, and answers it on a task of its own behind a slot of
//! its kind, so the loop goes straight back to polling.
//!
//! - **Chat turns** (and the listings of providers and models) share one slot: a
//!   long answer cannot overlap another, and a turn offered while one is being
//!   answered is left unclaimed for a later poll.
//! - **Pictures** (the `image` kind, [`crate::ai::tunnel_pictures`]) have a slot of
//!   their own: one picture at a time on this computer. A picture offered while
//!   another is being made is left unclaimed for a later poll. While a picture is
//!   drawn it posts a `progress` frame every five seconds, and its picture, or
//!   why there is none, comes in a `final` frame as a turn's answer does.
//!
//! So a chat turn that comes while a picture is drawn is answered meanwhile, and a
//! picture that comes while a chat turn is answered is drawn meanwhile. A row
//! whose claim is lost runs nothing, of either kind: the claim is atomic
//! server-side, and its slot is free again at once.
//!
//! A request the provider ends while it is answered (its requester cancelled it,
//! or it ran out of time) is stopped here too (contract §0.13). The provider says
//! so in its answer to the request's next frame, `409 {code: "cancelled"}`
//! ([`FrameRefused::Cancelled`]): a picture's request to the engines is dropped,
//! which ends their job; a chat's stream from its provider is dropped, which ends
//! the answer; nothing more is sent for the request, not a frame and not an
//! outcome; and its slot is free. Answered so to its last frame, there is nothing
//! left to stop. Any other refusal of a frame is what it always was: a frame lost.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::ai::chat_tools;
use crate::ai::e2e::{E2eIdentity, E2eSessions};
use crate::ai::providers::Capability;
use crate::ai::tunnel_pictures::{self, PictureError};
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

    /// The provider ended the request while it was being answered ([`FrameRefused::Cancelled`]). Never sealed into
    /// a frame: [`AiTunnel::finish`] sends nothing for it.
    fn cancelled() -> Self {
        Self::new(CANCELLED, "the provider cancelled the request")
    }

    fn is_cancelled(&self) -> bool {
        self.code == CANCELLED
    }
}

/// The code of [`TunnelError::cancelled`]: this desktop's own, and no code a requester is ever sent.
const CANCELLED: &str = "cancelled";

/// Why the provider did not take a frame.
#[derive(Debug, PartialEq, Eq)]
enum FrameRefused {
    /// The provider has ended the request itself (its requester cancelled it, or it ran out of time) and says so:
    /// `409 {code: "cancelled"}` (contract §0.13). An ordinary end, not trouble with the link: the work stops,
    /// nothing more is sent for the request, and its slot is free.
    Cancelled,
    /// Anything else. The frame is lost and the request goes on, as it always did.
    Other(String),
}

impl std::fmt::Display for FrameRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameRefused::Cancelled => f.write_str("the provider cancelled the request"),
            FrameRefused::Other(why) => f.write_str(why),
        }
    }
}

/// Whether a 409 of the provider's says it has ended the request: `{code: "cancelled"}` (contract §0.13). The
/// reply is read out either way.
async fn says_cancelled(refusal: reqwest::Response) -> bool {
    let said: Value = crate::link::net::read_json_async(refusal).await.unwrap_or(Value::Null);
    said.get("code").and_then(Value::as_str) == Some(CANCELLED)
}

/// The longest answer a chat request may ask for, in tokens: more than any model writes, so a larger number is a
/// mistake and is ignored as a missing one is ([`answer_limit`]).
const MAX_ANSWER_TOKENS: u64 = 1_000_000;

/// The answer's limit in tokens a sealed chat request asks for (`maxTokens`, contract §0.13), for the model's
/// `max_tokens`: a whole number from 1 to [`MAX_ANSWER_TOKENS`]. Anything else (none, zero, a negative, a fraction,
/// text, a number past the bound) is no limit of the requester's, and the provider's own applies as it did.
fn answer_limit(body: &Value) -> Option<u64> {
    body.get("maxTokens")
        .and_then(Value::as_u64)
        .filter(|n| (1..=MAX_ANSWER_TOKENS).contains(n))
}

impl From<PictureError> for TunnelError {
    fn from(e: PictureError) -> Self {
        Self::new(e.code(), e.message())
    }
}

/// How often a stream's accumulated text is sealed into a frame.
///
/// The model produces fragments far faster than a relay round trip, and one
/// POST per token would spend the whole turn on HTTP. Coalescing is invisible
/// to the reader — the browser appends whatever arrives — and still shows the
/// answer building rather than landing all at once.
const DELTA_FLUSH_INTERVAL: Duration = Duration::from_millis(220);
/// …or sooner once this much text is waiting (the ChatGPT stream).
const DELTA_FLUSH_BYTES: usize = 400;
/// Bound on the frames one turn may emit, so a runaway stream cannot hold the
/// lane open indefinitely. Of a provider's stream it bounds the chunks, each of
/// which was once a frame: an answer is as long as it could be before they were
/// gathered ([`Gathered`]), and no longer.
const MAX_STREAM_FRAMES: u64 = 4096;
/// How often word of a turn whose answer has not begun goes by itself (how far
/// the prompt has got, while a long one goes in): see `stream_registry`.
const BEFORE_THE_ANSWER_EVERY: Duration = Duration::from_secs(1);
/// The most text gathered into one frame of a provider's stream before it goes
/// without waiting for its interval: far under what a frame carries (256 KiB),
/// and far over what a model writes in an interval.
const GATHER_AT_MOST_BYTES: usize = 32 * 1024;

/// Everything one claimed turn needs to talk back to the provider.
///
/// Bundled because a streaming answer posts frames from several places, and
/// threading five parameters through each of them invites getting one wrong —
/// the instance id in particular, which every call must carry.
pub struct Lane<'a> {
    http: reqwest::Client,
    account: &'a LinkedAccount,
    spec: &'a DesktopAiSpec,
    instance: &'a str,
    id: &'a str,
}

impl Lane<'_> {
    /// The relayed request this lane belongs to — also the E2E session key.
    pub fn id(&self) -> &str {
        self.id
    }

    /// Poll the sealed inbound channel for frames after `since`.
    ///
    /// The cursor is the provider's own frame sequence, which is global rather
    /// than per-request, so it is opaque: advance it from what arrives and
    /// never assume it starts at one.
    pub async fn fetch_input(&self, since: u64) -> Result<Value, String> {
        let url = format!(
            "{}?since={since}&instanceId={}",
            crate::link::oauth::join(
                &self.account.base_url,
                &self.spec.input_path.replace("{id}", self.id),
            ),
            urlencode(self.instance),
        );
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.account.credential)
            .send()
            .await
            .map_err(|e| format!("could not poll for approvals: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            crate::link::net::drain_async(resp).await;
            return Err(format!("the approval poll failed: HTTP {status}"));
        }
        crate::link::net::read_json_async::<Value>(resp)
            .await
            .map_err(|e| format!("the approval poll returned nothing readable: {e}"))
    }

    async fn post_frame(&self, envelope: &str) -> Result<(), FrameRefused> {
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
            .map_err(|e| FrameRefused::Other(format!("could not post a frame: {e}")))?;
        let status = resp.status();
        // The one refusal whose words are read: the provider saying the request is over (contract §0.13).
        if status.as_u16() == 409 {
            if says_cancelled(resp).await {
                return Err(FrameRefused::Cancelled);
            }
            return Err(FrameRefused::Other("the lane refused a frame: HTTP 409".into()));
        }
        // Otherwise only the status is wanted; the rest is read out so the
        // connection can carry the next frame.
        crate::link::net::drain_async(resp).await;
        if !status.is_success() {
            return Err(FrameRefused::Other(format!("the lane refused a frame: HTTP {}", status.as_u16())));
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
    /// `chat`, `models`, `providers` or `image` (a picture, made in its own
    /// slot: [`AiTunnel::start_picture`]).
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
    codex_catalog_probe: OptionalProviderProbe,
    /// Where the engines are, when a test says (the outer `Option`); otherwise
    /// `http::engines_ui`, as the AI routes ask (`AiState::engines_ui`).
    engines: Option<Option<String>>,
    /// The client this lane keeps from poll to poll, so that a poll, the claim it
    /// leads to, the turn's frames and its completion share a connection instead
    /// of each making one. Its default timeout is what a turn's [`Lane`] has
    /// always had; the polls and the key publish set their own.
    http: crate::link::net::LaneClient<reqwest::Client>,
    /// This desktop's data folder: where the engines' settings name the age check's models (aceage).
    data_dir: std::path::PathBuf,
    /// The one picture this computer makes at a time (the site's rule too): a permit is held from a picture's claim
    /// until its outcome is reported, on the task that makes it, and goes back when that task ends, however it ends.
    picture_slot: Arc<tokio::sync::Semaphore>,
    /// The one other turn (a chat, or a listing) this computer answers at a time, beside the picture: held from its
    /// claim until its outcome is reported, on the task that answers it, as the picture slot is (contract §0.11).
    turn_slot: Arc<tokio::sync::Semaphore>,
    /// How often a picture being drawn says so: [`tunnel_pictures::PROGRESS_EVERY`] (a test's own, shorter).
    picture_progress: Duration,
}

/// An optional provider must not stall discovery of working local providers.
/// Keep at most one blocking status probe alive, even when its caller times out.
#[derive(Default)]
struct OptionalProviderProbe {
    running: Arc<AtomicBool>,
    connected: Arc<AtomicBool>,
}

impl OptionalProviderProbe {
    async fn check(&self, budget: Duration, probe: impl FnOnce() -> bool + Send + 'static) -> bool {
        if self.running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return self.connected.load(Ordering::Acquire);
        }
        let running = self.running.clone();
        let connected = self.connected.clone();
        let task = tokio::task::spawn_blocking(move || {
            struct Reset(Arc<AtomicBool>);
            impl Drop for Reset {
                fn drop(&mut self) { self.0.store(false, Ordering::Release); }
            }
            let _reset = Reset(running);
            let result = probe();
            connected.store(result, Ordering::Release);
            result
        });
        match tokio::time::timeout(budget, task).await {
            Ok(Ok(result)) => result,
            _ => self.connected.load(Ordering::Acquire),
        }
    }
}

impl AiTunnel {
    pub fn new(data_dir: &std::path::Path, sources: AiSources) -> Result<Arc<Self>, String> {
        Self::with_engines(data_dir, sources, None)
    }

    fn with_engines(data_dir: &std::path::Path, sources: AiSources, engines: Option<Option<String>>) -> Result<Arc<Self>, String> {
        let identity = E2eIdentity::load_or_create(data_dir)?;
        Ok(Arc::new(Self {
            identity,
            sessions: E2eSessions::new(),
            sources,
            published_marker: data_dir.join("desktop-e2e-published.json"),
            publish_note: std::sync::Mutex::new(None),
            codex_catalog_probe: OptionalProviderProbe::default(),
            engines,
            http: crate::link::net::LaneClient::new(build_client),
            data_dir: data_dir.to_path_buf(),
            picture_slot: Arc::new(tokio::sync::Semaphore::new(1)),
            turn_slot: Arc::new(tokio::sync::Semaphore::new(1)),
            picture_progress: tunnel_pictures::PROGRESS_EVERY,
        }))
    }

    /// Where the engines' control pages are (none: not running).
    fn engines_ui(&self) -> Option<String> {
        match &self.engines {
            Some(own) => own.clone(),
            None => crate::http::engines_ui(),
        }
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
            // Said once per reason, not every ten seconds while the provider is away.
            let mut last_warned: Option<String> = None;
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
                let polled = std::time::Instant::now();
                match tunnel.poll_cycle(&account, &spec, &instance).await {
                    Ok(handled) => {
                        if last_warned.take().is_some() {
                            log::info!("AI tunnel poll: working again");
                        }
                        if handled == 0 {
                            let pause = spec.idle_pause(polled.elapsed());
                            tunnel.http.rest(pause);
                            tokio::time::sleep(pause).await;
                        }
                    }
                    Err(e) => {
                        if last_warned.as_deref() != Some(e.as_str()) {
                            log::warn!("AI tunnel poll: {e}");
                            last_warned = Some(e);
                        }
                        // The try after a failure is on a new client, not one
                        // that has seen the trouble.
                        tunnel.http.start_afresh();
                        tokio::time::sleep(Duration::from_secs(spec.error_backoff_seconds)).await;
                    }
                }
            }
        });
    });
}

/// How long a request made through a turn's [`Lane`] may take: its frames, the
/// approvals it waits for, the tool catalogue and each tool run.
const TURN_TIMEOUT: Duration = Duration::from_secs(30);

fn build_client() -> Result<reqwest::Client, String> {
    crate::link::net::async_builder(crate::link::net::Keep::Between)
        .timeout(TURN_TIMEOUT)
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
                        && v.get("linkedAt").and_then(Value::as_str)
                            == Some(account.linked_at.to_rfc3339().as_str())
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
                    "linkedAt": account.linked_at.to_rfc3339(),
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
        let http = self.http.get()?;
        let resp = http
            .post(crate::link::oauth::join(&account.base_url, &spec.pubkey_path))
            .timeout(Duration::from_secs(20))
            .bearer_auth(&account.credential)
            .json(&json!({ "instanceId": instance, "publicKey": pubkey }))
            .send()
            .await
            .map_err(|e| format!("could not reach the AI lane: {e}"))?;
        if resp.status().is_success() {
            crate::link::net::drain_async(resp).await;
            return Ok(());
        }
        let status = resp.status().as_u16();
        let body: Value = crate::link::net::read_json_async(resp).await.unwrap_or(Value::Null);
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

    /// One long-poll and the turns it carried. Returns how many were handled: a
    /// picture counts once its claim is won and it is being made.
    async fn poll_cycle(
        self: &Arc<Self>,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
    ) -> Result<usize, String> {
        // A process that cannot hold the spent list cannot open what it claims, so it claims nothing: the row stays
        // for the process that can. (An error, for the loop's one line and its back-off.)
        self.identity.may_open_requests()?;
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
        let http = self.http.get()?;
        let resp = http
            .get(&url)
            // Generous over the server's wait so a long-poll that returns
            // exactly on time is not cut off by this side and retried needlessly.
            .timeout(Duration::from_secs(spec.wait_seconds + 15))
            .bearer_auth(&account.credential)
            .send()
            .await
            .map_err(|e| format!("could not reach the AI lane: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            // Asked to slow down: wait as long as the provider said before the lane's own back-off.
            if status.as_u16() == 429 {
                if let Some(wait) = crate::link::net::retry_after(resp.headers()) {
                    tokio::time::sleep(wait).await;
                }
            }
            let body: Value = crate::link::net::read_json_async(resp).await.unwrap_or(Value::Null);
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
        let reply: PendingReply = crate::link::net::read_json_async_up_to(resp, crate::link::net::MAX_LIST_BYTES)
            .await
            .map_err(|e| format!("the AI lane returned an unreadable batch: {e}"))?;

        let mut handled = 0usize;
        let mut failed = None;
        for request in reply.requests {
            // A picture is claimed here, in the batch's order, and made beside this loop, which goes straight on to
            // the next turn.
            if request.kind.as_deref() == Some(tunnel_pictures::KIND) {
                let id = request.id.clone();
                match self.start_picture(account, spec, instance, request).await {
                    Ok(true) => handled += 1,
                    // The slot is taken (the row is left for a later poll) or the claim was lost: nothing was served,
                    // and as with a turn the lane refused, this is not counted as work.
                    Ok(false) => {}
                    Err(e) => {
                        log::warn!("AI picture {id} could not be claimed: {e}");
                        failed.get_or_insert(e);
                        handled += 1;
                    }
                }
                continue;
            }
            // A chat turn (or a listing) the same way, behind a slot of its own (contract §0.11): a picture offered while
            // it is answered is claimed by the next poll, as a turn offered while a picture is drawn is.
            let id = request.id.clone();
            match self.start_turn(account, spec, instance, request).await {
                Ok(true) => handled += 1,
                // Taken, the lane is busy, or this computer's turn slot is: nothing was served. The row is still
                // offered while the lane is busy, so counting this as work would go straight round again, as fast as
                // the provider answers, until the lane comes free.
                Ok(false) => {}
                Err(e) => {
                    log::warn!("AI turn {id} could not be served: {e}");
                    failed.get_or_insert(e);
                    handled += 1;
                }
            }
        }
        // Polling that works while every turn fails (the provider half down)
        // must back off like a failed poll, not go straight round again.
        match failed {
            Some(e) if handled > 0 => Err(format!("{handled} AI turn(s) could not be served: {e}")),
            _ => Ok(handled),
        }
    }

    /// Claim one turn (a chat, or a listing of providers or models) and answer it on a task of its own, holding the
    /// turn slot until its terminal status is reported. True once it is being answered.
    ///
    /// False, with nothing claimed, while the slot is taken: a computer answers one turn at a time, and the site
    /// sends it one picture and one other request at a time (contract §0.11). Left unclaimed, the row stays the
    /// site's, offered again at the next poll and claimed once the slot is free. False too when the claim is lost,
    /// and then nothing of it runs.
    async fn start_turn(
        self: &Arc<Self>,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
        request: AiRequest,
    ) -> Result<bool, String> {
        let Ok(slot) = Arc::clone(&self.turn_slot).try_acquire_owned() else {
            return Ok(false);
        };
        let http = self.http.get()?;
        // (lost or refused, `slot` goes as this returns, and the slot is free again)
        if !self.claim(&http, account, spec, instance, &request.id).await? {
            return Ok(false);
        }
        let tunnel = Arc::clone(self);
        let (account, spec, instance) = (account.clone(), spec.clone(), instance.to_string());
        tokio::spawn(async move {
            let _slot = slot;
            let lane = Lane {
                http,
                account: &account,
                spec: &spec,
                instance: &instance,
                id: &request.id,
            };
            let outcome = tunnel.answer(&lane, &request).await;
            if let Err(e) = tunnel.finish(&lane, &request, outcome).await {
                log::warn!("AI turn {} could not be reported: {e}", request.id);
            }
        });
        Ok(true)
    }

    /// Take request `id`, exactly once: the claim is atomic on the provider's side. False when
    /// another got there first (409), the ordinary end of a lost race, after which nothing of the
    /// request may run.
    async fn claim(
        &self,
        http: &reqwest::Client,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
        id: &str,
    ) -> Result<bool, String> {
        let claim_url = crate::link::oauth::join(
            &account.base_url,
            &spec.claim_path.replace("{id}", id),
        );
        let claimed = http
            .post(&claim_url)
            .bearer_auth(&account.credential)
            .json(&json!({ "instanceId": instance }))
            .send()
            .await
            .map_err(|e| format!("could not claim: {e}"))?;
        // Only the status is wanted; the rest is read out so the connection can
        // carry what follows.
        let claim_status = claimed.status();
        crate::link::net::drain_async(claimed).await;
        if !claim_status.is_success() {
            // 409 is the ordinary "someone else got there first".
            return if claim_status.as_u16() == 409 {
                Ok(false)
            } else {
                Err(format!("claim refused: HTTP {}", claim_status.as_u16()))
            };
        }
        Ok(true)
    }

    /// Seal what a claimed request came to, post it as its last frame, and report it `done` or
    /// `failed`.
    ///
    /// A request the provider ended meanwhile (contract §0.13) gets neither: its work was stopped
    /// when a frame of it was answered so ([`TunnelError::cancelled`]), or its last frame is
    /// answered so here, where there is nothing left to stop. Either way that is an ordinary end.
    async fn finish(
        &self,
        lane: &Lane<'_>,
        request: &AiRequest,
        outcome: Result<Vec<u8>, TunnelError>,
    ) -> Result<(), String> {
        if matches!(&outcome, Err(e) if e.is_cancelled()) {
            self.ended_by_provider(request, "its work is stopped");
            return Ok(());
        }
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
            match lane.post_frame(&envelope).await {
                Ok(()) => {}
                Err(FrameRefused::Cancelled) => {
                    self.ended_by_provider(request, "it was already answered");
                    return Ok(());
                }
                Err(e) => log::warn!("AI turn {} final frame: {e}", request.id),
            }
        }
        let complete_url = crate::link::oauth::join(
            &lane.account.base_url,
            &lane.spec.complete_path.replace("{id}", &request.id),
        );
        // Exactly `done` or `failed`. Any other word is a 400, the row stays
        // claimed, and the requester waits out the TTL and is told it expired.
        let done = lane
            .http
            .post(&complete_url)
            .bearer_auth(&lane.account.credential)
            .json(&json!({ "instanceId": lane.instance, "status": status }))
            .send()
            .await
            .map_err(|e| format!("could not report the outcome: {e}"))?;
        self.sessions.drop_thread(&request.id);
        let done_status = done.status();
        if done_status.as_u16() == 409 {
            // The provider ended the request between its last frame and this (it answers the outcome of such a
            // request as it answers its frames): the same ordinary end.
            if says_cancelled(done).await {
                self.ended_by_provider(request, "it was already answered");
                return Ok(());
            }
            return Err("the AI lane refused the outcome: HTTP 409".into());
        }
        crate::link::net::drain_async(done).await;
        if !done_status.is_success() {
            return Err(format!(
                "the AI lane refused the outcome: HTTP {}",
                done_status.as_u16()
            ));
        }
        Ok(())
    }

    /// The provider ended a request this desktop had claimed ([`FrameRefused::Cancelled`]): nothing
    /// more is sent for it (no frame, no outcome), its session goes, and it is said once. Its slot is
    /// free as its task ends. `what`: what became of its work.
    fn ended_by_provider(&self, request: &AiRequest, what: &str) {
        self.sessions.drop_thread(&request.id);
        let kind = if request.kind.as_deref() == Some(tunnel_pictures::KIND) { "picture" } else { "turn" };
        log::info!("AI {kind} {}: cancelled by the provider; {what}, and nothing more is sent for it", request.id);
    }

    /// Claim a picture and make it on a task of its own, holding the picture slot until its outcome
    /// is reported. True once it is being made.
    ///
    /// False, with nothing claimed, while the slot is taken: the site sends one picture at a time
    /// to a computer too, so a second comes only when the two disagree for a while (a picture the
    /// site has given up on is still being drawn here). Left unclaimed it stays the site's, offered
    /// again at the next poll and claimed once the first is done. False too when the claim is lost,
    /// and then nothing of it runs.
    async fn start_picture(
        self: &Arc<Self>,
        account: &LinkedAccount,
        spec: &DesktopAiSpec,
        instance: &str,
        request: AiRequest,
    ) -> Result<bool, String> {
        let Ok(slot) = Arc::clone(&self.picture_slot).try_acquire_owned() else {
            return Ok(false);
        };
        let http = self.http.get()?;
        // (lost or refused, `slot` goes as this returns, and the slot is free again)
        if !self.claim(&http, account, spec, instance, &request.id).await? {
            return Ok(false);
        }
        let tunnel = Arc::clone(self);
        let (account, spec, instance) = (account.clone(), spec.clone(), instance.to_string());
        tokio::spawn(async move {
            let _slot = slot;
            let lane = Lane {
                http,
                account: &account,
                spec: &spec,
                instance: &instance,
                id: &request.id,
            };
            // Drawn, saying so every so often until it is done (each frame the site takes is news of the request,
            // and the site's answer to one is where it says the request is over).
            let outcome = {
                let making = tunnel.answer_picture(&request);
                tokio::pin!(making);
                let every = tunnel.picture_progress;
                let mut ticks = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
                ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tokio::select! {
                        outcome = &mut making => break outcome,
                        _ = ticks.tick() => {
                            if tunnel.post_progress(&lane).await {
                                // Cancelled: `making` goes as this block ends, and with it the request to the
                                // engines, which stop the job of a caller that has gone (their worker is ended
                                // at once, a model it was still loading included) or never start it.
                                break Err(TunnelError::cancelled());
                            }
                        }
                    }
                }
            };
            // (a cancelled one is said once, by `finish`)
            if let Some(e) = outcome.as_ref().err().filter(|e| !e.is_cancelled()) {
                log::info!("AI picture {}: {} ({})", request.id, e.code(), e.message());
            }
            if let Err(e) = tunnel.finish(&lane, &request, outcome).await {
                log::warn!("AI picture {} could not be reported: {e}", request.id);
            }
        });
        Ok(true)
    }

    /// Say that a picture is still being drawn ([`tunnel_pictures::progress_frame`]). Best effort:
    /// a frame that does not land costs the site a sign of life, not the picture. True when the
    /// site answers that the request is over ([`FrameRefused::Cancelled`]): the picture is not to
    /// be drawn any further.
    async fn post_progress(&self, lane: &Lane<'_>) -> bool {
        // (only once the request has opened: before that there is no session to seal with, and
        // nothing yet to say)
        let Ok(envelope) = self
            .sessions
            .seal_outbound(lane.id, tunnel_pictures::progress_frame().to_string().as_bytes())
        else {
            return false;
        };
        match lane.post_frame(&envelope).await {
            Ok(()) => false,
            Err(FrameRefused::Cancelled) => true,
            Err(e) => {
                log::warn!("AI picture {} progress frame: {e}", lane.id);
                false
            }
        }
    }

    /// Open a picture request and make the picture: the plaintext of its terminal frame.
    async fn answer_picture(&self, request: &AiRequest) -> Result<Vec<u8>, TunnelError> {
        let body = self.open_request(request)?;
        let asked = tunnel_pictures::PictureRequest::from_sealed(request.provider_id.as_deref(), &body)?;
        Ok(tunnel_pictures::make(self.engines_ui(), &asked, tunnel_pictures::PICTURE_TIMEOUT).await?)
    }

    /// Seal a typed failure so the requester learns WHY.
    ///
    /// When the request envelope never opened there is no session to count
    /// from, so this seals detached at outbound counter 0 — the browser has not
    /// seen an outbound frame for this turn either, so the counters agree.
    fn seal_error(&self, request: &AiRequest, code: &str, message: &str) -> Option<String> {
        // A picture's says it in the shape and the codes of the picture contract, whatever went wrong.
        let frame = if request.kind.as_deref() == Some(tunnel_pictures::KIND) {
            tunnel_pictures::error_frame(code, message)
        } else {
            json!({
                "v": 1,
                "type": "error",
                "kind": "error",
                "code": code,
                "message": message,
            })
        }
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
        // Counter 0 under this key once, ever. A garbled envelope naming a key the backend holds
        // back would otherwise get a reply of known text at the nonce the real answer then uses.
        if !self.identity.spend_request_key(&eph).unwrap_or(false) {
            return None;
        }
        crate::ai::e2e::seal_detached_envelope(
            &self.identity.secret_key_bytes(),
            &eph,
            crate::ai::e2e::DIR_DESKTOP_TO_BROWSER,
            0,
            frame.as_bytes(),
        )
        .ok()
    }

    /// Open a request's sealed envelope: its body, as JSON.
    fn open_request(&self, request: &AiRequest) -> Result<Value, TunnelError> {
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
        serde_json::from_slice(&plaintext)
            .map_err(|_| TunnelError::new("sealed_envelope_invalid", "the sealed body is not JSON"))
    }

    /// Open the sealed turn and produce the plaintext to seal back.
    async fn answer(&self, lane: &Lane<'_>, request: &AiRequest) -> Result<Vec<u8>, TunnelError> {
        let body = self.open_request(request)?;
        let eph_pub = request.eph_pub.as_deref().unwrap_or_default();
        if body.get("v").and_then(Value::as_u64) != Some(1) {
            return Err(TunnelError::new(
                "sealed_envelope_invalid",
                "the sealed body has no v:1 marker",
            ));
        }
        sealed_route_matches(request, &body)?;
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
            "chat" => self.chat(lane, provider_id, model.as_deref(), &body, eph_pub).await,
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
        // OAIY's own engine, offered when the engines answer with a language model
        // chosen, as `/api/ai/sources` offers it on this computer. Not in the store,
        // it was missing here, so FormLogic could not pick it through the relay. The
        // engines are asked within two seconds, like the ChatGPT probe below: a slow
        // or stopped engine must not hold up the list.
        if let Ok(Ok((_, model))) = tokio::time::timeout(
            Duration::from_secs(2),
            crate::ai::routes::engine_now_at(self.engines_ui()),
        )
        .await
        {
            if !model.is_empty() {
                rows.push(json!({
                    "id": crate::ai::routes::ENGINE_PROVIDER_ID,
                    "label": "OAIY engine",
                    "capabilities": ["chat"],
                    "enabled": true,
                    "model": model,
                }));
            }
        }
        // The managed ChatGPT connector, offered only when it is actually signed
        // in — listing it otherwise offers a guaranteed failure.
        let codex = self.sources.codex.clone();
        if self.codex_catalog_probe
            .check(Duration::from_secs(2), move || codex.status().connected)
            .await
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
        } else if provider_id == crate::ai::routes::ENGINE_PROVIDER_ID {
            // The engine's list needs no language model chosen, which answering a chat turn does
            // (`resolve`): engines that make only pictures list their picture models, each with its
            // `type`, and that is where the site finds the model an `image` request names.
            let (gateway, _) = crate::ai::routes::engine_now_at(self.engines_ui())
                .await
                .map_err(|e| TunnelError::new("provider_unavailable", format!("OAIY's engine cannot answer: {e}")))?;
            crate::ai::gateway::models(&crate::ai::routes::engine_provider(gateway, String::new()))
                .await
                .map_err(|e| TunnelError::new("provider_unavailable", e.message()))?
        } else {
            let (provider, _) = self.resolve(provider_id).await?;
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
        eph_pub: &str,
    ) -> Result<Vec<u8>, TunnelError> {
        // An age check's pictures read by this OAIY's own age model first, when the turn asks (`verify`): loaded in the
        // media worker for the check and gone when it ends, before the language model answers. Its readings go in
        // the answer's final frame. A turn whose provider cannot answer here is refused before the check, which can
        // take two minutes of this lane: the site takes an error frame as no answer (FormLogic refunds the try), so
        // readings run for one would be thrown away.
        let verification = match (body.get("verify"), body.get("messages").and_then(Value::as_array)) {
            (Some(ask), Some(messages)) => {
                self.can_answer(provider_id).await?;
                if is_codex(provider_id) {
                    self.codex_can_answer().await?;
                }
                Some(crate::ai::faceage::verify(&self.data_dir, ask, messages).await)
            }
            _ => None,
        };
        let frame = self.answer_chat(lane, provider_id, model, body, eph_pub).await?;
        Ok(match verification {
            Some(v) => crate::ai::faceage::with_verification(frame, v),
            None => frame,
        })
    }

    /// Whether a chat turn for `provider_id` can be answered here at all: asked before work only an answer uses.
    async fn can_answer(&self, provider_id: &str) -> Result<(), TunnelError> {
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
            return Ok(());
        }
        self.resolve(provider_id).await.map(|_| ())
    }

    /// Before an age check for a ChatGPT turn: whether ChatGPT can answer it, as far as Codex says. Signed out since
    /// the site chose it, or Codex gone from this computer, the answer fails after the check. Refused only on a plain
    /// answer: no account, or no Codex; a status slow to come, or an error of Codex's own, lets the turn go on, as
    /// before this was asked. Asked for an age check alone: other turns go to ChatGPT as they did.
    async fn codex_can_answer(&self) -> Result<(), TunnelError> {
        // (asked once at a time: a Codex that hangs held one more thread for each turn that asked, each waiting on the
        // session's lock behind the one before, and the answers behind them all; while one asks, the turn goes on)
        static ASKING: AtomicBool = AtomicBool::new(false);
        // (the last plain refusal and when: a turn that comes while one asks takes it while it is fresh, so two age-check
        // turns at once are not both let through to a check whose answer would fail)
        static REFUSED: std::sync::Mutex<Option<(std::time::Instant, &'static str)>> = std::sync::Mutex::new(None);
        let recent = || REFUSED.lock().ok().and_then(|r| *r).filter(|(at, _)| at.elapsed() < Duration::from_secs(30)).map(|(_, why)| why);
        if ASKING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return match recent() {
                Some(why) => Err(TunnelError::new("provider_unavailable", why)),
                None => Ok(()),
            };
        }
        let codex = self.sources.codex.clone();
        let status = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::task::spawn_blocking(move || {
                struct Done;
                impl Drop for Done {
                    fn drop(&mut self) {
                        ASKING.store(false, Ordering::Release);
                    }
                }
                let _done = Done;
                codex.status()
            }),
        )
        .await;
        let refused = match status {
            Ok(Ok(status)) if !status.available => Some("ChatGPT (Codex) is not available on this computer"),
            Ok(Ok(status)) if !status.connected && status.detail.is_none() => Some("ChatGPT is not signed in on this computer"),
            _ => None,
        };
        if let Ok(mut last) = REFUSED.lock() {
            *last = refused.map(|why| (std::time::Instant::now(), why));
        }
        match refused {
            Some(why) => Err(TunnelError::new("provider_unavailable", why)),
            None => Ok(()),
        }
    }

    async fn answer_chat(
        &self,
        lane: &Lane<'_>,
        provider_id: &str,
        model: Option<&str>,
        body: &Value,
        eph_pub: &str,
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
        // How long the answer may be, when the requester says (contract §0.13): the model's
        // `max_tokens`, so an answer that runs on ends there instead of holding the lane for
        // minutes. This desktop adds no limit of its own; a provider's own still applies.
        let max_tokens = answer_limit(body);
        if let Some(n) = max_tokens {
            chat_body["max_tokens"] = json!(n);
        }

        // A turn carrying a grant may act on the account. Without one — or with
        // a catalogue that is empty or unreachable — this is byte-for-byte the
        // plain path below, so a provider with no tools loses nothing.
        let grant = body
            .get("toolGrant")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|g| !g.is_empty());
        if let (Some(grant), Some(catalog_path), Some(execute_path)) = (
            grant,
            lane.spec.tools_catalog_path.as_deref(),
            lane.spec.tools_execute_path.as_deref(),
        ) {
            let tools = chat_tools::catalog(
                &lane.http,
                &lane.account.base_url,
                &lane.account.credential,
                catalog_path,
            )
            .await;
            if !tools.is_empty() {
                return self
                    .chat_with_tools(lane, provider_id, model, &messages, ToolRun {
                        grant,
                        execute_path,
                        tools,
                        confirm: body.get("toolMode").and_then(Value::as_str) == Some("confirm"),
                        eph_pub: eph_pub.to_string(),
                        max_tokens,
                    })
                    .await;
            }
        }

        if is_codex(provider_id) {
            self.can_answer(provider_id).await?;
            return self.stream_codex(lane, chat_body).await;
        }

        let (provider, engine) = self.resolve(provider_id).await?;
        // The engine answers with the model chosen in Engines, as its local route does: the
        // gateway puts the provider's model in when the request names none.
        if engine {
            if let Some(obj) = chat_body.as_object_mut() {
                obj.remove("model");
            }
        }
        if crate::ai::gateway::streamable(&provider) {
            chat_body["stream"] = json!(true);
            return self.stream_registry(lane, &provider, chat_body).await;
        }
        let completion = crate::ai::gateway::chat(&provider, chat_body)
            .await
            .map_err(|e| TunnelError::new("upstream_error", e.message()))?;
        Ok(final_with_completion(completion))
    }

    /// Run the bounded tool loop for one turn.
    ///
    /// Non-streaming by design: the loop has to READ each reply before any of
    /// it is user-visible, because the reply may be a tool call rather than an
    /// answer. Streaming a call's fenced block to the user would show them the
    /// machinery instead of the result.
    async fn chat_with_tools(
        &self,
        lane: &Lane<'_>,
        provider_id: &str,
        model: Option<&str>,
        messages: &[Value],
        run: ToolRun<'_>,
    ) -> Result<Vec<u8>, TunnelError> {
        let (native, model) = if is_codex(provider_id) {
            (None, model)
        } else {
            let (p, engine) = self.resolve(provider_id).await?;
            // Only the OpenAI dialect carries tool schemas out and tool calls
            // back; anything translated would drop them on one leg or the other.
            // The engine answers with the model chosen in Engines (see `chat`).
            ((p.protocol == crate::ai::providers::Protocol::OpenAi).then_some(p), if engine { None } else { model })
        };

        let mut convo: Vec<Value> = messages.to_vec();
        // Extra turns the prompt-only transport has to carry itself, since it
        // has no structured history to append to.
        let mut prompted_extra: Vec<String> = Vec::new();
        let mut grant_dead = false;
        let mut input_cursor: u64 = 0;

        for round in 0..chat_tools::MAX_TOOL_ROUNDS {
            let offer_tools = chat_tools::tools_available(round, grant_dead);
            let calls = match &native {
                Some(provider) => {
                    let mut request = json!({ "messages": convo });
                    if let Some(m) = model {
                        request["model"] = json!(m);
                    }
                    if let Some(n) = run.max_tokens {
                        request["max_tokens"] = json!(n);
                    }
                    if offer_tools {
                        request["tools"] = json!(chat_tools::as_openai_tools(&run.tools));
                    }
                    let completion = crate::ai::gateway::chat(provider, request)
                        .await
                        .map_err(|e| TunnelError::new("upstream_error", e.message()))?;
                    let message = completion
                        .pointer("/choices/0/message")
                        .cloned()
                        .unwrap_or(Value::Null);
                    let calls = if offer_tools {
                        chat_tools::parse_native(&message)
                    } else {
                        Vec::new()
                    };
                    if calls.is_empty() {
                        return Ok(final_with_completion(completion));
                    }
                    // The assistant's own turn has to go back in, or the tool
                    // replies below answer a message the model never sees. With
                    // the calls that are answered, and only them: a call past the
                    // limit, or one without a name, left in it has no tool reply,
                    // and an OpenAI-style API refuses the next request (400).
                    let mut message = message.clone();
                    message["tool_calls"] = Value::Array(
                        calls
                            .iter()
                            .map(|c| json!({ "id": c.provider_id, "type": "function", "function": { "name": c.name, "arguments": c.input.to_string() } }))
                            .collect(),
                    );
                    convo.push(message);
                    calls
                }
                None => {
                    let mut prompt = if offer_tools {
                        chat_tools::preamble(&run.tools)
                    } else {
                        chat_tools::plain_answer_instruction().to_string()
                    };
                    prompt.push_str("\n\n");
                    prompt.push_str(&chat_tools::render_conversation(&convo));
                    for extra in &prompted_extra {
                        prompt.push_str("\n\n");
                        prompt.push_str(extra);
                    }
                    let mut request = json!({
                        "messages": [{ "role": "user", "content": prompt }],
                    });
                    if let Some(m) = model {
                        request["model"] = json!(m);
                    }
                    let codex = self.sources.codex.clone();
                    let completion =
                        tokio::task::spawn_blocking(move || codex.chat_as(&request, None))
                            .await
                            .map_err(|e| TunnelError::new("upstream_error", e.to_string()))?
                            .map_err(|e| TunnelError::new(codex_code(&e), e.message()))?;
                    let text = completion
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let call = if offer_tools {
                        chat_tools::parse_prompted(&text, &run.tools)
                    } else {
                        None
                    };
                    match call {
                        None => return Ok(final_with_completion(completion)),
                        Some(c) => vec![c],
                    }
                }
            };

            for call in calls {
                self.seal_and_post(lane, &chat_tools::call_frame(&call)).await?;

                if run.confirm {
                    self.seal_and_post(lane, &chat_tools::proposal_frame(&call, lane.id()))
                        .await?;
                    let verdict = chat_tools::wait_for_approval(
                        lane,
                        &self.sessions,
                        &self.identity,
                        &run.eph_pub,
                        &call.id,
                        &mut input_cursor,
                    )
                    .await;
                    if let chat_tools::Approval::Denied(reason) = verdict {
                        // A denial is an ordinary outcome: the model is told and
                        // carries on. Failing the turn would punish the user for
                        // saying no.
                        let outcome = chat_tools::Executed::Failed {
                            error: format!("not run — {reason}"),
                            terminal: false,
                        };
                        self.seal_and_post(lane, &chat_tools::result_frame(&call, &outcome))
                            .await?;
                        push_result(&native, &mut convo, &mut prompted_extra, &call, &outcome);
                        continue;
                    }
                }

                let outcome = chat_tools::execute(
                    &lane.http,
                    &lane.account.base_url,
                    &lane.account.credential,
                    run.execute_path,
                    run.grant,
                    &call,
                )
                .await;
                if let chat_tools::Executed::Failed { terminal: true, .. } = outcome {
                    // The provider will refuse every later call identically;
                    // retrying would be dishonest noise.
                    grant_dead = true;
                }
                self.seal_and_post(lane, &chat_tools::result_frame(&call, &outcome))
                    .await?;
                push_result(&native, &mut convo, &mut prompted_extra, &call, &outcome);
            }
        }

        // Unreachable in practice — the last round is offered no tools, so it
        // answers — but a loop that could fall out of the bottom silently must
        // still say something honest.
        Err(TunnelError::new(
            "upstream_error",
            "the assistant kept calling tools without answering",
        ))
    }

    /// Seal one activity frame and post it.
    async fn seal_and_post(&self, lane: &Lane<'_>, frame: &Value) -> Result<(), TunnelError> {
        let envelope = self
            .sessions
            .seal_outbound(lane.id(), frame.to_string().as_bytes())
            .map_err(|e| TunnelError::new(e.code(), e.message()))?;
        match lane.post_frame(&envelope).await {
            Ok(()) => Ok(()),
            // The request is over: no further tool is run and the model is not asked again.
            Err(FrameRefused::Cancelled) => Err(TunnelError::cancelled()),
            // A dropped activity frame costs the user a progress line, not the
            // answer — the turn continues.
            Err(e) => {
                log::warn!("AI turn {} activity frame: {e}", lane.id());
                Ok(())
            }
        }
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

    /// Stream a registry provider's answer (OAIY's engine among them), its chunks
    /// gathered into a few frames a second.
    ///
    /// A model writes a chunk a token, fifty and more a second, and a frame each
    /// was a request each to the provider: 282 in five seconds of one answer,
    /// against the 600 a minute a FormLogic link's key may make, so one long
    /// answer had every other request of the link refused. The chunks are now
    /// joined ([`Gathered`]): the first text goes at once, so the reader sees the
    /// answer start, and what follows goes every [`DELTA_FLUSH_INTERVAL`], the
    /// last of it before the final frame. No text is lost by it: a chunk that
    /// cannot be joined to others goes as it came, in its place. Before the
    /// first text there is only word of the turn, which goes with that text, or
    /// by itself every [`BEFORE_THE_ANSWER_EVERY`] while the text is awaited.
    ///
    /// A frame's chunk is nested under `delta`, in the shape the provider sent.
    /// Spreading it at the frame's top level instead is silently dropped by the
    /// reader — no error, just an answer that never appears.
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
        // The provider's chunks read, and the frames made of them.
        let (mut chunks, mut frames) = (0u64, 0u64);
        // The chunks waiting to go as one frame, and when the last frame went (the stream's start, until one has).
        let mut waiting = Gathered::default();
        let mut last_frame = std::time::Instant::now();
        // Whether a frame with text in it has gone: until then, the first text does not wait.
        let mut begun = false;
        // How long what is waiting may wait since the last frame: an interval, once there is text. Before any,
        // only word of the turn is waiting (who speaks, how far the prompt has got), which nobody reads as it
        // comes: it goes with the first text, or by itself once a second while the answer is still to begin, so
        // that the provider hears of the turn and can say it was cancelled.
        let pace = |waiting: &Gathered, begun: bool| if begun || waiting.has_text() { DELTA_FLUSH_INTERVAL } else { BEFORE_THE_ANSWER_EVERY };
        loop {
            // What is waiting goes when its time comes, whether or not the provider sends more meanwhile (a model
            // that pauses mid-answer does not keep the last words back).
            let next = if waiting.is_empty() {
                stream.next().await
            } else {
                let due_in = pace(&waiting, begun).saturating_sub(last_frame.elapsed());
                match tokio::time::timeout(due_in, stream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        begun |= waiting.has_text();
                        self.post_delta(lane, waiting.chunk(), &mut frames).await?;
                        last_frame = std::time::Instant::now();
                        continue;
                    }
                }
            };
            let Some(chunk) = next else { break };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    // (what was read is not lost with the stream)
                    self.post_gathered(lane, &mut waiting, &mut frames).await?;
                    return Err(TunnelError::new("upstream_error", format!("stream read failed: {e}")));
                }
            };
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
                    chunks += 1;
                    if chunks > MAX_STREAM_FRAMES {
                        self.post_gathered(lane, &mut waiting, &mut frames).await?;
                        return Err(TunnelError::new(
                            "upstream_error",
                            "the provider's stream exceeded the frame cap",
                        ));
                    }
                    if !waiting.take(&delta) {
                        // Not text to join to other text (a tool call, a second choice): what is waiting goes
                        // first, then this as it came, so everything stays in the provider's order.
                        self.post_gathered(lane, &mut waiting, &mut frames).await?;
                        self.post_delta(lane, delta, &mut frames).await?;
                        last_frame = std::time::Instant::now();
                    }
                }
            }
            // The first text at once, so the reader sees the answer start; after it, a frame an interval (or
            // sooner, should more be waiting than a frame ought to carry).
            if !waiting.is_empty() && ((!begun && waiting.has_text()) || waiting.is_full() || last_frame.elapsed() >= pace(&waiting, begun)) {
                begun |= waiting.has_text();
                self.post_delta(lane, waiting.chunk(), &mut frames).await?;
                last_frame = std::time::Instant::now();
            }
        }
        // The stream is over: the last of it goes before the final frame.
        self.post_gathered(lane, &mut waiting, &mut frames).await?;
        if chunks == 0 {
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

    /// Post what is waiting of a provider's stream, if anything is.
    async fn post_gathered(&self, lane: &Lane<'_>, waiting: &mut Gathered, frames: &mut u64) -> Result<(), TunnelError> {
        if waiting.is_empty() {
            return Ok(());
        }
        self.post_delta(lane, waiting.chunk(), frames).await
    }

    /// Seal one chunk of a provider's stream (one of its own, or several gathered into one) as a delta frame, and
    /// post it.
    async fn post_delta(&self, lane: &Lane<'_>, chunk: Value, frames: &mut u64) -> Result<(), TunnelError> {
        *frames += 1;
        let frame = json!({ "v": 1, "type": "delta", "kind": "delta", "delta": chunk });
        let envelope = self
            .sessions
            .seal_outbound(lane.id, frame.to_string().as_bytes())
            .map_err(|e| TunnelError::new(e.code(), e.message()))?;
        match lane.post_frame(&envelope).await {
            Ok(()) => Ok(()),
            // The request is over. The provider's stream goes as this returns, and a
            // server whose reader has gone stops writing the answer (OAIY's engine does).
            Err(FrameRefused::Cancelled) => Err(TunnelError::cancelled()),
            // A dropped frame leaves a gap the reader can see in its
            // own text; abandoning the turn over it would be worse.
            Err(e) => {
                log::warn!("AI turn {} delta {frames}: {e}", lane.id);
                Ok(())
            }
        }
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
        match lane.post_frame(&envelope).await {
            Ok(()) => Ok(()),
            // The request is over: nothing more of the answer is read or sent. (ChatGPT's own turn
            // has no stop: it runs to its end on its thread, unheard.)
            Err(FrameRefused::Cancelled) => Err(TunnelError::cancelled()),
            Err(e) => {
                log::warn!("AI turn {} delta {streamed}: {e}", lane.id);
                Ok(())
            }
        }
    }

    /// Fail closed: an unknown, disabled or keyless provider is refused by name
    /// rather than silently substituted for whatever else is configured.
    ///
    /// `oaiy-engine` is OAIY's own engine, asked of the engines now as the local
    /// AI routes ask (`resolve_chat_provider`): it is not in the store. Looked up
    /// there alone, the relay refused the engine this desktop lists to FormLogic
    /// ("no provider named \"oaiy-engine\""), so a computer whose only AI is its
    /// engine could not be used through the relay at all. The `bool` says the
    /// engine is the one answering, which answers with the model chosen in
    /// Engines whatever the turn names.
    async fn resolve(&self, provider_id: &str) -> Result<(crate::ai::providers::AiProvider, bool), TunnelError> {
        if provider_id == crate::ai::routes::ENGINE_PROVIDER_ID {
            let (gateway, model) = crate::ai::routes::engine_now_at(self.engines_ui())
                .await
                .map_err(|e| TunnelError::new("provider_unavailable", format!("OAIY's engine cannot answer: {e}")))?;
            if model.is_empty() {
                return Err(TunnelError::new(
                    "provider_unavailable",
                    "OAIY's engine has no language model chosen: choose one in Engines",
                ));
            }
            return Ok((crate::ai::routes::engine_provider(gateway, model), true));
        }
        self.resolve_stored(provider_id).map(|provider| (provider, false))
    }

    /// A provider of the store, held to the rules above.
    fn resolve_stored(&self, provider_id: &str) -> Result<crate::ai::providers::AiProvider, TunnelError> {
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

/// Everything the tool loop needs that the plain path does not.
struct ToolRun<'a> {
    grant: &'a str,
    execute_path: &'a str,
    tools: Vec<chat_tools::Tool>,
    confirm: bool,
    /// The browser's per-request key, needed to OPEN its approval frames.
    eph_pub: String,
    /// The requester's limit on an answer ([`answer_limit`]): each reply of the model in the loop is held to it.
    max_tokens: Option<u64>,
}

/// Feed one tool's outcome back to the model, in whichever form its transport
/// understands: a `role:"tool"` message natively, or plain text for a
/// prompt-only model that has no such role.
fn push_result(
    native: &Option<crate::ai::providers::AiProvider>,
    convo: &mut Vec<Value>,
    prompted_extra: &mut Vec<String>,
    call: &chat_tools::ToolCall,
    outcome: &chat_tools::Executed,
) {
    let rendered = chat_tools::result_for_model(call, outcome);
    match native {
        Some(_) => convo.push(json!({
            "role": "tool",
            // Every tool_call must receive a reply carrying its id, or the
            // next request is malformed and the provider refuses the round.
            "tool_call_id": call.provider_id,
            "name": call.name,
            "content": rendered,
        })),
        None => prompted_extra.push(rendered),
    }
}

/// The fields of a streamed chunk's delta that are pieces of a longer text, joined in order when chunks are
/// gathered. Any other field of a delta (`role`) is a fact, of which the newest is kept.
const DELTA_TEXT_FIELDS: [&str; 4] = ["content", "reasoning_content", "reasoning", "refusal"];

/// Chunks of a provider's streamed answer (OpenAI's `chat.completion.chunk`) gathered into one of the same shape,
/// to go as one frame: a reader that joins each frame's `choices[].delta.content` reads the same text from one
/// gathered chunk as from the chunks it was made of.
///
/// - The delta's text fields ([`DELTA_TEXT_FIELDS`]) are joined in order.
/// - Of everything else the newest is kept, a null never replacing a value: the chunk's own fields (`id`, `model`,
///   `usage`), the choice's (`finish_reason`, which only the last chunk says) and the delta's (`role`).
/// - A chunk with no choices (`choices: []`) is no part of the answer but word of how the turn is getting on:
///   what was used (OpenAI's last chunk), or from OAIY's engine how far the prompt has got (`oaiy_progress`, sent
///   as it goes) and how much the model has thought (`oaiy_thinking`). Its fields are the chunk's own, so only
///   the newest of each goes on, with the text gathered beside it or by itself.
#[derive(Default)]
struct Gathered {
    /// The chunk's own fields, but `choices`.
    top: serde_json::Map<String, Value>,
    /// Its one choice's fields, but `delta`.
    choice: serde_json::Map<String, Value>,
    delta: serde_json::Map<String, Value>,
    /// Whether any of its chunks had a choice (else it goes with `choices: []`, as they came).
    spoke: bool,
    /// How many chunks are in it, and how many bytes of text.
    chunks: usize,
    text_bytes: usize,
}

impl Gathered {
    fn is_empty(&self) -> bool {
        self.chunks == 0
    }

    fn has_text(&self) -> bool {
        self.text_bytes > 0
    }

    fn is_full(&self) -> bool {
        self.text_bytes >= GATHER_AT_MOST_BYTES
    }

    /// Add `chunk`, if it can be joined to others with nothing lost: an object with no choices (word of the turn,
    /// see above), or with exactly one, of index 0, whose delta holds only text and nulls (so no tool call, which
    /// comes in pieces of its own kind) and whose other fields are plain values (so no per-token `logprobs`).
    /// False leaves it out, for the caller to send as it came: anything else, an object with no `choices` at all
    /// among them (an error, or a side note of a provider's own).
    fn take(&mut self, chunk: &Value) -> bool {
        let Some(top) = chunk.as_object() else { return false };
        let choice = match top.get("choices").and_then(Value::as_array).map(Vec::as_slice) {
            Some([]) => None,
            Some([choice]) => Some(choice),
            _ => return false,
        };
        let parts = match choice {
            None => None,
            Some(choice) => {
                let Some(choice) = choice.as_object() else { return false };
                let Some(delta) = choice.get("delta").and_then(Value::as_object) else { return false };
                let plain = |v: &Value| !v.is_array() && !v.is_object();
                if !delta.values().all(|v| v.is_string() || v.is_null())
                    || !choice.iter().all(|(key, v)| key == "delta" || plain(v))
                    || choice.get("index").is_some_and(|i| i.as_u64() != Some(0))
                {
                    return false;
                }
                Some((choice, delta))
            }
        };
        for (key, value) in top.iter().filter(|(key, _)| *key != "choices") {
            newest(&mut self.top, key, value);
        }
        self.chunks += 1;
        let Some((choice, delta)) = parts else { return true };
        self.spoke = true;
        for (key, value) in choice.iter().filter(|(key, _)| *key != "delta") {
            newest(&mut self.choice, key, value);
        }
        for (key, value) in delta {
            let text = DELTA_TEXT_FIELDS.contains(&key.as_str());
            let joined = text
                && match (value.as_str(), self.delta.get_mut(key)) {
                    (Some(piece), Some(Value::String(so_far))) => {
                        so_far.push_str(piece);
                        true
                    }
                    _ => false,
                };
            if !joined {
                newest(&mut self.delta, key, value);
            }
            if text {
                self.text_bytes += value.as_str().map_or(0, str::len);
            }
        }
        true
    }

    /// What was gathered, as one chunk, leaving this empty.
    fn chunk(&mut self) -> Value {
        let Gathered { mut top, mut choice, delta, spoke, .. } = std::mem::take(self);
        let choices = if spoke {
            choice.insert("delta".into(), Value::Object(delta));
            json!([choice])
        } else {
            json!([])
        };
        top.insert("choices".into(), choices);
        Value::Object(top)
    }
}

/// Keep `value` under `key`, unless it is a null and something is there already.
fn newest(into: &mut serde_json::Map<String, Value>, key: &str, value: &Value) {
    if !value.is_null() || !into.contains_key(key) {
        into.insert(key.to_string(), value.clone());
    }
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

/// Flatten a model catalogue into the plain `[{id, name?, type?}]` array the web
/// app renders. It keeps object entries only, so an OpenAI `{data:[…]}` body
/// passed through verbatim leaves the dropdown empty with nothing explaining why.
///
/// `type` is what kind of model it is, as OAIY's engines say (`image`, `video`,
/// `speech`; their language models say none), kept so the site can offer the
/// picture models for an `image` request and the others for chat.
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
            if let Some(kind) = entry
                .get("type")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && s.len() <= 32)
            {
                out["type"] = json!(kind);
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

/// The row's `kind` and `providerId` are routing plaintext, and the browser seals both. A backend
/// that rewrote them chose who reads a sealed prompt: one kept on this computer's own engine could
/// be sent to a cloud provider on the owner's key.
fn sealed_route_matches(request: &AiRequest, body: &Value) -> Result<(), TunnelError> {
    let kind = request.kind.as_deref().unwrap_or("chat");
    let provider = request.provider_id.as_deref().unwrap_or("");
    if body.get("kind").and_then(Value::as_str) == Some(kind)
        && body.get("providerId").and_then(Value::as_str) == Some(provider)
    {
        return Ok(());
    }
    Err(TunnelError::new("sealed_envelope_invalid", "the sealed turn does not match its row"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: &str, provider: &str, eph_pub: &str) -> AiRequest {
        serde_json::from_value(json!({ "requestId": "turn-1", "kind": kind, "providerId": provider, "ephPub": eph_pub })).unwrap()
    }

    #[test]
    fn a_row_cannot_send_a_sealed_turn_elsewhere() {
        let sealed = json!({ "v": 1, "kind": "chat", "providerId": "oaiy-engine" });
        assert!(sealed_route_matches(&row("chat", "oaiy-engine", "k"), &sealed).is_ok());
        for moved in [row("chat", "openai", "k"), row("models", "oaiy-engine", "k")] {
            assert_eq!(sealed_route_matches(&moved, &sealed).unwrap_err().code(), "sealed_envelope_invalid");
        }
        // A body with no route is not one this desktop's browsers seal.
        assert!(sealed_route_matches(&row("chat", "oaiy-engine", "k"), &json!({ "v": 1 })).is_err());
    }

    #[test]
    fn the_providers_list_keeps_its_placeholder_route() {
        // The browser asks for the list under the provider "all", sealed and on the row alike.
        let sealed = json!({ "v": 1, "kind": "providers", "providerId": "all" });
        assert!(sealed_route_matches(&row("providers", "all", "k"), &sealed).is_ok());
    }

    #[test]
    fn an_error_is_sealed_to_a_key_once() {
        // Whatever the backend knows the text of, it must not get twice at one nonce, nor before
        // the real answer: a garbled envelope naming a held-back key came first.
        let (tunnel, dir) = tunnel_in("seal-error");
        let secret = crypto_box::SecretKey::from([0xB7u8; 32]);
        let eph = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, secret.public_key().to_bytes());
        let request = row("chat", "oaiy-engine", &eph);
        assert!(tunnel.seal_error(&request, "invalid_request", "first").is_some());
        assert!(tunnel.seal_error(&request, "invalid_request", "second").is_none());
        let envelope = crate::ai::e2e::seal_detached_envelope(&[0xB7u8; 32], &tunnel.identity.public_key_bytes(), crate::ai::e2e::DIR_BROWSER_TO_DESKTOP, 0, b"{}").unwrap();
        let refused = tunnel.sessions.open_inbound(&tunnel.identity, "turn-2", &eph, &envelope).unwrap_err();
        assert!(refused.message().contains("used before"), "{}", refused.message());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn slow_optional_provider_does_not_block_catalog_or_duplicate_probes() {
        let probe = OptionalProviderProbe::default();
        let (release, wait) = std::sync::mpsc::channel();
        let first = probe.check(Duration::from_millis(10), move || {
            wait.recv_timeout(Duration::from_secs(2)).is_ok()
        }).await;
        assert!(!first);
        assert!(probe.running.load(Ordering::Acquire));
        let second = probe.check(Duration::from_millis(10), || {
            panic!("a timed-out probe must not be duplicated")
        }).await;
        assert!(!second);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while probe.running.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await.unwrap();
        assert!(probe.connected.load(Ordering::Acquire));
        assert!(!probe.check(Duration::from_secs(1), || false).await);
        assert!(!probe.connected.load(Ordering::Acquire));
    }

    /// A tunnel's data folder, gone when it is dropped. A poll holds the folder's spent list for the life of the
    /// process, and Windows removes no open file, so that goes first.
    struct TunnelDir(std::path::PathBuf);

    impl std::ops::Deref for TunnelDir {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl AsRef<std::path::Path> for TunnelDir {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TunnelDir {
        fn drop(&mut self) {
            crate::ai::e2e::let_go_of_spent_list(&self.0);
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A tunnel over a fresh data folder, with nothing to answer a turn with: these
    /// tests are about what it does on the wire.
    fn tunnel_in(tag: &str) -> (Arc<AiTunnel>, TunnelDir) {
        let dir = std::env::temp_dir().join(format!("oaiy-tunnel-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        (AiTunnel::new(&dir, sources).unwrap(), TunnelDir(dir))
    }

    /// A tunnel told where OAIY's engines are (none: nowhere), whatever `http` says.
    fn tunnel_with_engines(tag: &str, engines: Option<String>) -> (Arc<AiTunnel>, TunnelDir) {
        let dir = std::env::temp_dir().join(format!("oaiy-tunnel-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        (AiTunnel::with_engines(&dir, sources, Some(engines)).unwrap(), TunnelDir(dir))
    }

    #[tokio::test]
    async fn the_relay_lists_oaiys_engine_and_its_models() {
        // FormLogic picks a provider from this list and then asks for its models: the engine,
        // which /api/ai/sources offers on this computer, was missing from it, and its models
        // were refused as "no provider named \"oaiy-engine\"".
        let engines = crate::ai::routes::tests::fake_engines().await;
        let (tunnel, dir) = tunnel_with_engines("engine-list", Some(engines.ui.clone()));
        let listed: Value = serde_json::from_slice(&tunnel.list_providers().await).unwrap();
        let rows: Vec<&Value> = listed["providers"].as_array().unwrap().iter().filter(|r| r["id"] == "oaiy-engine").collect();
        assert_eq!(
            rows,
            [&json!({ "id": "oaiy-engine", "label": "OAIY engine", "capabilities": ["chat"], "enabled": true, "model": "flash" })],
            "the engine, bare, once, with the model chosen in Engines: {listed}"
        );
        let models: Value = serde_json::from_slice(&tunnel.list_models("oaiy-engine").await.unwrap()).unwrap();
        assert_eq!(models["models"][0]["id"], "flash", "{models}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_relayed_chat_reaches_oaiys_engine_with_the_model_chosen_in_engines() {
        // As the local route answers (chat_through_the_engine_provider_reaches_the_engines_and_
        // answers_with_the_chosen_model): the turn's own model name is not the engine's to use.
        let engines = crate::ai::routes::tests::fake_engines().await;
        let (tunnel, dir) = tunnel_with_engines("engine-chat", Some(engines.ui.clone()));
        // The engine answers in one piece, so nothing is posted to the account on this path.
        let account = linked("http://127.0.0.1:9".into(), "flk_ai");
        let spec = descriptor::builtin().remove(0).desktop_ai.unwrap();
        let lane = Lane { http: reqwest::Client::new(), account: &account, spec: &spec, instance: "oaiy-test", id: "turn-1" };
        let body = json!({ "v": 1, "messages": [{ "role": "user", "content": "hi" }] });
        let answer = tunnel.chat(&lane, "oaiy-engine", Some("a-model-nobody-chose"), &body, "unused").await.unwrap();
        let answer = String::from_utf8(answer).unwrap();
        assert!(answer.contains("hello from the engine"), "{answer}");
        {
            let asked = engines.asked.lock().unwrap();
            assert_eq!(asked.len(), 1, "one call reached the engines' gateway");
            assert_eq!(asked[0].1["model"], "flash", "the model chosen in Engines, not the one the turn named");
            assert_eq!(asked[0].1["messages"][0]["content"], "hi");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn with_no_engines_running_the_relay_offers_no_engine_and_says_so() {
        let (tunnel, dir) = tunnel_with_engines("engine-none", None);
        let listed: Value = serde_json::from_slice(&tunnel.list_providers().await).unwrap();
        assert!(listed["providers"].as_array().unwrap().iter().all(|r| r["id"] != "oaiy-engine"), "{listed}");
        let refused = tunnel.list_models("oaiy-engine").await.unwrap_err();
        assert_eq!(refused.code(), "provider_unavailable");
        assert!(refused.message.contains("OAIY's engine cannot answer"), "{}", refused.message);
        // A name the store does not hold is refused by that name, as before.
        let unknown = tunnel.list_models("nowhere").await.unwrap_err();
        assert!(unknown.message.contains("no provider named \"nowhere\""), "{}", unknown.message);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An age check runs for a turn only when its provider can answer it: the check takes up to two minutes of the lane,
    /// and the site takes an error frame as no answer, so it was run, and thrown away, for a provider this OAIY lacks.
    #[tokio::test]
    async fn an_age_check_is_not_run_for_a_turn_no_provider_here_can_answer() {
        let (tunnel, dir) = tunnel_in("verify-unknown");
        // An age check that would take thirty seconds: its files are there and its worker never answers.
        let engines = dir.join("engines");
        std::fs::create_dir_all(&engines).unwrap();
        let worker = engines.join(if cfg!(windows) { "slow.bat" } else { "slow.sh" });
        std::fs::write(&worker, if cfg!(windows) { "@ping -n 30 127.0.0.1 > nul\r\n" } else { "#!/bin/sh\nsleep 30\n" }).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        std::fs::write(engines.join("model.safetensors"), b"x").unwrap();
        std::fs::write(engines.join("yunet.onnx"), b"x").unwrap();
        std::fs::write(
            engines.join("oaiy-studio.json"),
            serde_json::to_vec(&json!({ "media": { "worker": worker, "faceage": { "model": "model.safetensors", "detector": "yunet.onnx" } } })).unwrap(),
        )
        .unwrap();
        assert!(crate::ai::faceage::files(&dir).is_ok(), "the check would run");

        let account = linked("http://127.0.0.1:9".into(), "flk_ai");
        let spec = descriptor::builtin().remove(0).desktop_ai.unwrap();
        let lane = Lane { http: reqwest::Client::new(), account: &account, spec: &spec, instance: "oaiy-test", id: "turn-1" };
        let body = json!({ "v": 1, "verify": { "task": "face-age", "images": [0] }, "messages": [{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,AAAA" } },
        ]}] });
        let started = std::time::Instant::now();
        let refused = tunnel.chat(&lane, "nowhere", None, &body, "unused").await.unwrap_err();
        assert!(refused.message.contains("no provider named \"nowhere\""), "{}", refused.message);
        assert!(started.elapsed() < Duration::from_secs(10), "refused before the check: {:?}", started.elapsed());
        // A live-call alias, which never serves a queued chat turn: refused the same way.
        let alias = crate::ai::codex::LiveCallAlias::ReasoningLow.id();
        let refused = tunnel.chat(&lane, alias, None, &body, "unused").await.unwrap_err();
        assert_eq!(refused.code(), "provider_unavailable");
        assert!(started.elapsed() < Duration::from_secs(10), "a live-call alias too: {:?}", started.elapsed());
        // ChatGPT with no one signed in (this tunnel's Codex home is new), or no Codex on this computer, which
        // stream_codex would refuse after the check.
        let started = std::time::Instant::now();
        let refused = tunnel.chat(&lane, crate::ai::codex::CODEX_PROVIDER_ID, None, &body, "unused").await.unwrap_err();
        assert_eq!(refused.code(), "provider_unavailable", "{}", refused.message);
        assert!(refused.message.contains("ChatGPT"), "{}", refused.message);
        assert!(started.elapsed() < Duration::from_secs(25), "refused before the thirty-second check: {:?}", started.elapsed());
    }

    fn linked(base: String, credential: &str) -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: base,
            credential: credential.into(),
            account_id: None,
            account_name: None,
            granted_scopes: None,
            linked_at: chrono::Utc::now(),
            instance_id: Some("oaiy-test".into()),
        }
    }

    /// Another process on the folder holds the spent list. This one used to claim the turn and then fail it, a lost
    /// turn for the person, where the holder would have answered: it now asks the provider for nothing at all.
    #[tokio::test]
    async fn a_lane_that_cannot_hold_the_spent_list_asks_the_provider_for_nothing() {
        use crate::link::testkit::{Provider, Reply};
        let (tunnel, dir) = tunnel_in("held");
        let holder = std::fs::File::create(dir.join("desktop-e2e-spent.lock")).unwrap();
        fs2::FileExt::try_lock_exclusive(&holder).unwrap();
        let server = Provider::start(|_| Reply::ok(r#"{"requests":[]}"#));
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        let refused = tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap_err();
        assert!(refused.contains("another OAIY process"), "{refused}");
        assert!(server.lines().is_empty(), "{:?}", server.lines());

        // The holder gone, the lane polls again by itself.
        drop(holder);
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0);
        assert_eq!(server.lines().len(), 1);
    }

    #[tokio::test]
    async fn the_key_publish_and_the_polls_share_one_connection() {
        // A poll every 25 seconds used to make a connection, and do a TLS
        // handshake, each time; the client is now kept from poll to poll.
        use crate::link::testkit::{Provider, Reply};
        let (tunnel, dir) = tunnel_in("reuse");
        let server = Provider::start(|req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(r#"{"requests":[]}"#)
            } else {
                // The answer to a publish is not read, and this one is too big to
                // have come in with its headers.
                Reply::ok(&format!(r#"{{"pad":"{}"}}"#, "x".repeat(200_000)))
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        tunnel.publish(&account, &spec, "oaiy-test", "the-public-key").await.unwrap();
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0);
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0);

        assert_eq!(
            server.lines(),
            [
                "POST /api/v1/desktop-ai/pubkey",
                "GET /api/v1/desktop-ai/pending?wait=25000&limit=8&instanceId=oaiy-test",
                "GET /api/v1/desktop-ai/pending?wait=25000&limit=8&instanceId=oaiy-test",
            ]
        );
        assert_eq!(server.connections(), 1, "{:?}", server.lines());
        assert!(server.requests().iter().all(|r| r.header("authorization") == Some("Bearer flk_ai")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_lane_waits_after_an_empty_poll_as_long_as_its_descriptor_says() {
        // The lane's own loop, not a function beside it: a provider that asks for a
        // three second pause is not polled again for three seconds. (Every lane
        // used to wait half a second, or the rest of two seconds if the provider
        // cut its poll short, and there was nowhere to say otherwise.)
        use crate::link::testkit::{Provider, Reply};
        let server = Provider::start(|_| Reply::ok(r#"{"requests":[]}"#));
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "ai-slow", |d| {
            let lane = d.desktop_ai.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.idle_pause_ms = 3_000;
        });
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        spawn(store.clone(), sources);
        let polls = server.wait_for("/api/v1/desktop-ai/pending", 3, Duration::from_secs(30));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        for pair in polls.windows(2) {
            let gap = pair[1].at - pair[0].at;
            assert!(gap >= Duration::from_millis(2_800) && gap < Duration::from_secs(6), "{gap:?}");
        }
    }

    #[test]
    fn a_turn_it_could_not_claim_waits_like_an_empty_poll() {
        // While the lane is busy the row is still offered and the claim refused. Counted as work,
        // that went straight round: a poll and a claim, as fast as the provider answered, for as
        // long as the lane stayed busy.
        use crate::link::testkit::{Provider, Reply};
        let row = json!({
            "requests": [{
                "requestId": "1f0c2a1e-0000-4000-8000-000000000001",
                "kind": "chat",
                "providerId": "x",
                "ephPub": "AAAA",
                "envelope": "AAAA",
            }]
        })
        .to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&row)
            } else {
                Reply::status(409, r#"{"code":"lane_busy"}"#)
            }
        });
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "ai-busy", |d| {
            let lane = d.desktop_ai.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.idle_pause_ms = 1_500;
        });
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        spawn(store.clone(), sources);
        let polls = server.wait_for("/api/v1/desktop-ai/pending", 3, Duration::from_secs(30));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        for pair in polls.windows(2) {
            let gap = pair[1].at - pair[0].at;
            assert!(gap >= Duration::from_millis(1_200), "polled again after {gap:?}");
        }
    }

    #[test]
    fn a_long_pause_after_an_empty_poll_is_not_spent_holding_a_connection_open() {
        // The pause of a provider that cannot hold polls (a hold of a second, a
        // pause of five) is time it means to be free of the lane, and a connection
        // held through it, for four seconds or more, would not be.
        use crate::link::testkit::{Provider, Reply};
        let server = Provider::start(|_| Reply::ok(r#"{"requests":[]}"#));
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "ai-rest", |d| {
            let lane = d.desktop_ai.as_mut().unwrap();
            lane.wait_seconds = 1;
            lane.idle_pause_ms = 5_000;
        });
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        spawn(store.clone(), sources);
        server.wait_for("/api/v1/desktop-ai/pending", 1, Duration::from_secs(30));
        let held = server.try_closed_after_reply(0, Duration::from_secs(3));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        assert!(held.is_some_and(|held| held < Duration::from_secs(2)), "the connection was held open: {held:?}");
    }

    #[test]
    fn a_poll_that_failed_is_followed_by_one_on_a_new_connection() {
        // A lane starts its client afresh after a failed poll, as it was when every poll
        // made its own client. The second poll comes a second after the first, well
        // inside what a connection is kept for, so on a client that had been kept it
        // would arrive on the same connection.
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::{AtomicBool, Ordering};
        let failed = AtomicBool::new(false);
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") && !failed.swap(true, Ordering::SeqCst) {
                Reply::status(500, r#"{"message":"down for a moment"}"#)
            } else {
                Reply::ok(r#"{"requests":[]}"#)
            }
        });
        let (store, dir) = crate::link::testkit::linked_to(&server.base, "ai-afresh", |d| {
            d.desktop_ai.as_mut().unwrap().error_backoff_seconds = 1;
        });
        let sources = AiSources {
            providers: crate::ai::providers::new_handle(),
            codex: crate::ai::codex::new_handle(&dir),
        };
        spawn(store.clone(), sources);
        let polls = server.wait_for("/api/v1/desktop-ai/pending", 2, Duration::from_secs(30));
        crate::link::testkit::stop_lane(&store);
        let _ = std::fs::remove_dir_all(dir);
        assert_eq!(polls.iter().map(|p| p.conn).collect::<Vec<_>>(), [0, 1], "{:?}", server.lines());
    }

    #[tokio::test]
    async fn the_tunnel_gives_each_account_only_its_own_credential() {
        use crate::link::testkit::{Provider, Reply};
        let (tunnel, dir) = tunnel_in("credentials");
        let empty = || Provider::start(|_| Reply::ok(r#"{"requests":[]}"#));
        let (a, b) = (empty(), empty());
        let spec = descriptor::builtin().remove(0).desktop_ai.unwrap();
        for (server, credential) in [(&a, "flk_alpha"), (&b, "flk_beta"), (&a, "flk_gamma")] {
            tunnel.poll_cycle(&linked(server.base.clone(), credential), &spec, "oaiy-test").await.unwrap();
        }
        let bearers = |server: &Provider| -> Vec<String> {
            server.requests().iter().map(|r| r.header("authorization").unwrap_or("none").to_string()).collect()
        };
        assert_eq!(bearers(&a), ["Bearer flk_alpha", "Bearer flk_gamma"]);
        assert_eq!(bearers(&b), ["Bearer flk_beta"]);
        let _ = std::fs::remove_dir_all(dir);
    }

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
    fn a_models_type_is_kept_so_the_site_can_list_picture_models() {
        // OAIY's engines say what each model is; their language models say nothing.
        let engines = json!({ "object": "list", "data": [
            { "id": "nanbeige4.3-33b", "object": "model", "owned_by": "oaiy", "loaded": true },
            { "id": "flux-3-klein", "object": "model", "owned_by": "oaiy-studio", "type": "image", "loaded": false },
            { "id": "ltx-2.5", "object": "model", "owned_by": "oaiy-studio", "type": "video", "loaded": false },
            { "id": "odd", "type": 7 },
        ]});
        assert_eq!(
            normalize_models(&engines),
            vec![
                json!({ "id": "nanbeige4.3-33b" }),
                json!({ "id": "flux-3-klein", "type": "image" }),
                json!({ "id": "ltx-2.5", "type": "video" }),
                json!({ "id": "odd" }),
            ]
        );
    }

    // ---- pictures (the `image` kind) ----

    use crate::ai::tunnel_pictures::tests::{fake_engines, sealed, Setup};
    use base64::Engine as _;

    const B64: base64::engine::general_purpose::GeneralPurpose = base64::engine::general_purpose::STANDARD;

    /// `body` sealed to `tunnel` with the requester's key `seed`, as a row carries it: `(ephPub, envelope)`.
    fn sealed_to(tunnel: &AiTunnel, seed: u8, body: &Value) -> (String, String) {
        let public = crypto_box::SecretKey::from([seed; 32]).public_key().to_bytes();
        let envelope = crate::ai::e2e::seal_detached_envelope(
            &[seed; 32],
            &tunnel.identity.public_key_bytes(),
            crate::ai::e2e::DIR_BROWSER_TO_DESKTOP,
            0,
            body.to_string().as_bytes(),
        )
        .unwrap();
        (B64.encode(public), envelope)
    }

    /// A pending row of the provider's.
    fn pending_row(id: &str, kind: &str, (eph, envelope): &(String, String)) -> Value {
        json!({ "requestId": id, "kind": kind, "providerId": "oaiy-engine", "ephPub": eph, "envelope": envelope })
    }

    /// The frames `tunnel` posted for request `id`, in order, opened with the requester's key `seed`: what the
    /// requester reads. Each is checked to be at the next counter, as a reader holds them to.
    fn frames_of(tunnel: &AiTunnel, server: &crate::link::testkit::Provider, id: &str, seed: u8) -> Vec<Value> {
        let posted: Vec<_> = server
            .requests()
            .into_iter()
            .filter(|r| r.line() == format!("POST /api/v1/desktop-ai/{id}/frames"))
            .collect();
        assert!(!posted.is_empty(), "no frame for {id}: {:?}", server.lines());
        posted
            .iter()
            .enumerate()
            .map(|(counter, frame)| {
                let envelope = serde_json::from_str::<Value>(&frame.body).unwrap()["envelope"].as_str().unwrap().to_string();
                let (nonce, ct) = crate::ai::e2e::unpack_envelope(&envelope).unwrap();
                assert_eq!(nonce, crate::ai::e2e::frame_nonce(crate::ai::e2e::DIR_DESKTOP_TO_BROWSER, counter as u64));
                let plain = crate::ai::e2e::open_detached(
                    &[seed; 32],
                    &tunnel.identity.public_key_bytes(),
                    crate::ai::e2e::DIR_DESKTOP_TO_BROWSER,
                    &B64.encode(nonce),
                    &B64.encode(ct),
                )
                .unwrap();
                serde_json::from_slice(&plain).unwrap()
            })
            .collect()
    }

    /// The last frame posted for request `id`: its answer.
    fn frame_of(tunnel: &AiTunnel, server: &crate::link::testkit::Provider, id: &str, seed: u8) -> Value {
        frames_of(tunnel, server, id, seed).pop().unwrap()
    }

    /// The status request `id` was reported with.
    fn completed_as(server: &crate::link::testkit::Provider, id: &str) -> Option<String> {
        server
            .requests()
            .into_iter()
            .find(|r| r.line() == format!("POST /api/v1/desktop-ai/{id}/complete"))
            .map(|r| serde_json::from_str::<Value>(&r.body).unwrap()["status"].as_str().unwrap().to_string())
    }

    /// Wait (yielding, so the picture's task runs) until `done`, or fail after 20 seconds.
    async fn until(what: &str, done: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while !done() {
            assert!(std::time::Instant::now() < deadline, "waited 20 seconds for {what}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The contract's concurrency, one way: a picture takes a minute, and the chat turn after it in the same batch is
    /// answered meanwhile. The engines here make the picture only once the chat turn has been reported, so a lane that
    /// served them one after the other would wait on itself (the engines give up waiting after twenty seconds; this
    /// test has ten). A second picture offered while the first is being made is left unclaimed, and claimed at the
    /// next poll once the slot is free.
    #[tokio::test]
    async fn a_chat_turn_is_answered_while_a_picture_is_being_made() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        use std::sync::Mutex;
        let lines: Arc<Mutex<Vec<String>>> = Arc::default();
        let chat_reported = {
            let lines = lines.clone();
            move || lines.lock().unwrap().iter().any(|l| l == "POST /api/v1/desktop-ai/chat-1/complete")
        };
        // (the picture was at the engines while the chat turn was still open)
        let overlapped = Arc::new(AtomicBool::new(false));
        let hold = {
            let overlapped = overlapped.clone();
            move || {
                let reported = chat_reported();
                if !reported {
                    overlapped.store(true, Ordering::SeqCst);
                }
                reported
            }
        };
        let engines = fake_engines(Setup {
            picture: crate::ai::tunnel_pictures::tests::detailed_png(1024, 1024),
            chat_delay: Duration::from_millis(500),
            hold: Some(Arc::new(hold)),
            ..Setup::default()
        })
        .await;
        let (mut tunnel, _dir) = tunnel_with_engines("picture-beside-chat", Some(engines.ui.clone()));
        // (says it is drawing every tenth of a second, not every fifteen)
        Arc::get_mut(&mut tunnel).unwrap().picture_progress = Duration::from_millis(100);

        let first = sealed_to(&tunnel, 0x11, &sealed(json!({ "providerId": "oaiy-engine" })));
        let chat = sealed_to(&tunnel, 0x22, &json!({ "v": 1, "kind": "chat", "providerId": "oaiy-engine", "messages": [{ "role": "user", "content": "hi" }] }));
        let second = sealed_to(&tunnel, 0x33, &sealed(json!({ "providerId": "oaiy-engine", "seed": null, "size": "512x512" })));
        let batch = json!({ "requests": [pending_row("picture-1", "image", &first), pending_row("chat-1", "chat", &chat), pending_row("picture-2", "image", &second)] }).to_string();
        let later = json!({ "requests": [pending_row("picture-2", "image", &second)] }).to_string();
        let polls = Arc::new(AtomicUsize::new(0));
        let server = {
            let (lines, polls) = (lines.clone(), polls.clone());
            Provider::start(move |req| {
                lines.lock().unwrap().push(req.line());
                if req.target.starts_with("/api/v1/desktop-ai/pending") {
                    Reply::ok(if polls.fetch_add(1, Ordering::SeqCst) == 0 { &batch } else { &later })
                } else {
                    Reply::ok("{}")
                }
            })
        };
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        let started = std::time::Instant::now();
        let handled = tokio::time::timeout(Duration::from_secs(10), tunnel.poll_cycle(&account, &spec, "oaiy-test"))
            .await
            .expect("the poll waited on what it claimed")
            .unwrap();
        assert_eq!(handled, 2, "the picture being made and the chat turn being answered: {:?}", server.lines());
        assert!(!server.lines().iter().any(|l| l.contains("picture-2")), "the second picture is left unclaimed: {:?}", server.lines());
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        assert_eq!(completed_as(&server, "chat-1").as_deref(), Some("done"));

        until("the picture", || completed_as(&server, "picture-1").is_some()).await;
        assert!(started.elapsed() < Duration::from_secs(10), "the chat turn waited for the picture: {:?}", started.elapsed());
        assert!(overlapped.load(Ordering::SeqCst), "the picture was being made while the chat turn was answered");
        let order = server.lines();
        let at = |line: &str| order.iter().position(|l| l == line).unwrap();
        assert!(at("POST /api/v1/desktop-ai/chat-1/complete") < at("POST /api/v1/desktop-ai/picture-1/complete"), "{order:?}");
        assert!(at("POST /api/v1/desktop-ai/picture-1/claim") < at("POST /api/v1/desktop-ai/chat-1/claim"), "claimed in the batch's order: {order:?}");
        assert_eq!(completed_as(&server, "picture-1").as_deref(), Some("done"));

        let answer = frames_of(&tunnel, &server, "chat-1", 0x22);
        assert_eq!(answer.len(), 1, "a chat turn's frames are as they were: {answer:?}");
        assert!(answer[0].to_string().contains("hello from the engine"), "{answer:?}");
        // While it was drawn the picture said so, in frames of their own kind, and then came in chat's final kind.
        let mut frames = frames_of(&tunnel, &server, "picture-1", 0x11);
        let made = frames.pop().unwrap();
        assert!(!frames.is_empty(), "no progress frame while the picture was drawn");
        assert!(frames.iter().all(|f| *f == crate::ai::tunnel_pictures::progress_frame()), "{frames:?}");
        assert_eq!((made["kind"].as_str(), made["type"].as_str()), (Some("final"), Some("final")));
        let picture = &made["picture"];
        assert_eq!((picture["mimeType"].as_str(), picture["seed"].as_u64(), picture["model"].as_str()), (Some("image/jpeg"), Some(12345), Some("flux-3-klein")));
        let bytes = B64.decode(picture["data"].as_str().unwrap()).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!(u64::from(decoded.width()), picture["width"].as_u64().unwrap());
        assert!(made.to_string().len() <= crate::ai::e2e::MAX_ENVELOPE_PLAINTEXT_BYTES);

        // The slot is free (once the picture's task has read the answer to its outcome and ended): the next poll
        // claims the second picture, whose seed is this computer's to choose.
        until("the picture slot", || tunnel.picture_slot.available_permits() == 1).await;
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the second picture", || completed_as(&server, "picture-2").is_some()).await;
        let second = frame_of(&tunnel, &server, "picture-2", 0x33);
        let sent = engines.pictures_asked();
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert_eq!(second["picture"]["seed"], sent[1].body["seed"], "the seed chosen is the one reported");
        assert!(sent.iter().all(|a| a.body["incognito"] == true && a.incognito_header.as_deref() == Some("1")));
    }

    /// The contract's concurrency, the other way (§0.11): a picture offered while a chat turn is being answered is
    /// claimed by the next poll and drawn meanwhile. The engines here answer the chat turn only once the picture has
    /// been reported, so a lane that answered the chat turn before it polled again would wait on itself. A second
    /// chat turn offered meanwhile is left unclaimed (one turn at a time), and claimed once the first is reported.
    #[tokio::test]
    async fn a_picture_is_drawn_while_a_chat_turn_is_answered() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        use std::sync::Mutex;
        let lines: Arc<Mutex<Vec<String>>> = Arc::default();
        let picture_reported = {
            let lines = lines.clone();
            move || lines.lock().unwrap().iter().any(|l| l == "POST /api/v1/desktop-ai/picture-1/complete")
        };
        let engines = fake_engines(Setup {
            picture: crate::ai::tunnel_pictures::tests::detailed_png(1024, 1024),
            chat_hold: Some(Arc::new(picture_reported)),
            ..Setup::default()
        })
        .await;
        let (mut tunnel, _dir) = tunnel_with_engines("chat-beside-picture", Some(engines.ui.clone()));
        Arc::get_mut(&mut tunnel).unwrap().picture_progress = Duration::from_millis(100);

        let turn = |seed: u8| sealed_to(&tunnel, seed, &json!({ "v": 1, "kind": "chat", "providerId": "oaiy-engine", "messages": [{ "role": "user", "content": "hi" }] }));
        let (chat_1, chat_2) = (turn(0x61), turn(0x62));
        let picture = sealed_to(&tunnel, 0x63, &sealed(json!({ "providerId": "oaiy-engine" })));
        let batches = [
            json!({ "requests": [pending_row("chat-1", "chat", &chat_1)] }).to_string(),
            json!({ "requests": [pending_row("chat-2", "chat", &chat_2), pending_row("picture-1", "image", &picture)] }).to_string(),
            json!({ "requests": [pending_row("chat-2", "chat", &chat_2)] }).to_string(),
        ];
        let polls = Arc::new(AtomicUsize::new(0));
        let server = {
            let (lines, polls) = (lines.clone(), polls.clone());
            Provider::start(move |req| {
                lines.lock().unwrap().push(req.line());
                if req.target.starts_with("/api/v1/desktop-ai/pending") {
                    Reply::ok(&batches[polls.fetch_add(1, Ordering::SeqCst).min(2)])
                } else {
                    Reply::ok("{}")
                }
            })
        };
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());
        let cycle = || tokio::time::timeout(Duration::from_secs(5), tunnel.poll_cycle(&account, &spec, "oaiy-test"));

        let started = std::time::Instant::now();
        assert_eq!(cycle().await.expect("the poll waited on the chat turn").unwrap(), 1, "{:?}", server.lines());
        // The chat turn is being answered (its engine holds it): the next poll claims the picture, and not chat-2.
        assert_eq!(cycle().await.expect("the poll waited on the chat turn").unwrap(), 1, "{:?}", server.lines());
        assert!(completed_as(&server, "chat-1").is_none(), "the chat turn is still being answered: {:?}", server.lines());
        assert!(!server.lines().iter().any(|l| l.contains("chat-2")), "a second chat turn is left unclaimed: {:?}", server.lines());
        assert_eq!(tunnel.turn_slot.available_permits(), 0);

        until("the picture", || completed_as(&server, "picture-1").is_some()).await;
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        assert!(started.elapsed() < Duration::from_secs(10), "the picture waited for the chat turn: {:?}", started.elapsed());
        assert_eq!((completed_as(&server, "picture-1").as_deref(), completed_as(&server, "chat-1").as_deref()), (Some("done"), Some("done")));
        let order = server.lines();
        let at = |line: &str| order.iter().position(|l| l == line).unwrap();
        assert!(at("POST /api/v1/desktop-ai/chat-1/claim") < at("POST /api/v1/desktop-ai/picture-1/claim"), "{order:?}");
        assert!(at("POST /api/v1/desktop-ai/picture-1/claim") < at("POST /api/v1/desktop-ai/chat-1/complete"), "the picture was claimed while the chat turn was open: {order:?}");
        assert!(at("POST /api/v1/desktop-ai/picture-1/complete") < at("POST /api/v1/desktop-ai/chat-1/complete"), "{order:?}");
        let answer = frames_of(&tunnel, &server, "chat-1", 0x61);
        assert!(answer.last().unwrap().to_string().contains("hello from the engine"), "{answer:?}");
        let drawn = frame_of(&tunnel, &server, "picture-1", 0x63);
        assert_eq!(drawn["picture"]["mimeType"], "image/jpeg", "{drawn}");

        // The turn slot is free once chat-1 is reported (its task gives it back): chat-2 is claimed and answered.
        until("the turn slot", || tunnel.turn_slot.available_permits() == 1).await;
        assert_eq!(cycle().await.unwrap().unwrap(), 1);
        until("the second chat turn", || completed_as(&server, "chat-2").is_some()).await;
        assert_eq!(completed_as(&server, "chat-2").as_deref(), Some("done"));
        assert_eq!(engines.pictures_asked().len(), 1);
        assert_eq!(engines.asked.lock().unwrap().iter().filter(|a| a.route == "chat").count(), 2, "each chat turn answered once");
    }

    /// A chat turn whose claim another computer won is never answered, as a picture is never made: the engines are not
    /// asked, nothing is posted for it, and the turn slot is free again.
    #[tokio::test]
    async fn a_chat_turn_whose_claim_is_lost_is_never_answered() {
        use crate::link::testkit::{Provider, Reply};
        let engines = fake_engines(Setup::default()).await;
        let (tunnel, _dir) = tunnel_with_engines("chat-lost", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x71, &json!({ "v": 1, "kind": "chat", "providerId": "oaiy-engine", "messages": [{ "role": "user", "content": "hi" }] }));
        let batch = json!({ "requests": [pending_row("chat-1", "chat", &row)] }).to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else if req.target.ends_with("/claim") {
                Reply::status(409, r#"{"code":"already_claimed"}"#)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0, "a lost claim is not work done");
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(engines.asked.lock().unwrap().is_empty(), "{:?}", engines.asked.lock().unwrap());
        assert_eq!(tunnel.turn_slot.available_permits(), 1, "the slot is free again");
        assert!(!server.lines().iter().any(|l| l.ends_with("/frames") || l.ends_with("/complete")), "{:?}", server.lines());
    }

    /// What the site answers a frame of a request it has ended (contract §0.13).
    const CANCELLED_BY_THE_SITE: &str = r#"{"error":true,"code":"cancelled","message":"This request was cancelled."}"#;

    /// A chat turn's sealed body.
    fn chat_turn() -> Value {
        json!({ "v": 1, "kind": "chat", "providerId": "oaiy-engine", "messages": [{ "role": "user", "content": "hi" }] })
    }

    /// How many frames `tunnel`'s side posted for request `id`, taken or not.
    fn frames_posted(server: &crate::link::testkit::Provider, id: &str) -> usize {
        server.lines().iter().filter(|l| **l == format!("POST /api/v1/desktop-ai/{id}/frames")).count()
    }

    /// Contract §0.13: the site answers a frame of a request it has ended `409 {code: "cancelled"}`. Answered so to
    /// a progress frame, a picture stops being drawn: the request to the engines is dropped (they end the job of a
    /// caller that has gone), nothing more is sent for it, not a frame and not an outcome, and the picture slot is
    /// free for the next picture.
    #[tokio::test]
    async fn a_picture_the_site_cancels_is_stopped_at_the_engines_and_its_slot_is_free() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        // The engines keep the first picture until the test lets go (they would wait twenty seconds for it).
        let released = Arc::new(AtomicBool::new(false));
        let hold = {
            let released = released.clone();
            move || released.load(Ordering::SeqCst)
        };
        let engines = fake_engines(Setup { hold: Some(Arc::new(hold)), ..Setup::default() }).await;
        let (mut tunnel, _dir) = tunnel_with_engines("picture-cancelled", Some(engines.ui.clone()));
        Arc::get_mut(&mut tunnel).unwrap().picture_progress = Duration::from_millis(100);

        let first = sealed_to(&tunnel, 0x81, &sealed(json!({ "providerId": "oaiy-engine" })));
        let second = sealed_to(&tunnel, 0x82, &sealed(json!({ "providerId": "oaiy-engine" })));
        let batches = [
            json!({ "requests": [pending_row("picture-1", "image", &first)] }).to_string(),
            json!({ "requests": [pending_row("picture-2", "image", &second)] }).to_string(),
        ];
        let (polls, frames) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batches[polls.fetch_add(1, Ordering::SeqCst).min(1)])
            } else if req.line() == "POST /api/v1/desktop-ai/picture-1/frames" && frames.fetch_add(1, Ordering::SeqCst) >= 1 {
                // Its first progress frame was taken; the site has ended the request since, and says so to the next.
                Reply::status(409, CANCELLED_BY_THE_SITE)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        let started = std::time::Instant::now();
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the engines to see their caller gone", || engines.abandoned.load(Ordering::SeqCst) == 1).await;
        until("the picture slot", || tunnel.picture_slot.available_permits() == 1).await;
        assert!(started.elapsed() < Duration::from_secs(5), "stopped at the second progress frame, not when the engines gave up: {:?}", started.elapsed());
        // (time for anything more to be sent)
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(frames_posted(&server, "picture-1"), 2, "one frame taken, one answered cancelled, none after: {:?}", server.lines());
        assert!(completed_as(&server, "picture-1").is_none(), "no outcome is reported for it: {:?}", server.lines());
        assert!(tunnel.sessions.seal_outbound("picture-1", b"more").is_err(), "its session is gone: nothing more can be sealed for it");
        assert_eq!(engines.pictures_asked().len(), 1);

        // The slot is free: the next picture is claimed and drawn as any other.
        released.store(true, Ordering::SeqCst);
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the next picture", || completed_as(&server, "picture-2").is_some()).await;
        assert_eq!(completed_as(&server, "picture-2").as_deref(), Some("done"));
        assert_eq!(frame_of(&tunnel, &server, "picture-2", 0x82)["picture"]["mimeType"], "image/png");
        assert_eq!(engines.abandoned.load(Ordering::SeqCst), 1, "only the cancelled one was dropped");
    }

    /// The same for a chat turn, answered so to a frame of its streamed answer: the provider's stream is dropped,
    /// which stops the answer being written (as OAIY's engine stops for a reader that has gone), nothing more is
    /// sent, and the turn slot is free.
    #[tokio::test]
    async fn a_chat_turn_the_site_cancels_stops_being_written_and_its_slot_is_free() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        // An answer of 400 pieces, 10 ms apart: four seconds of writing.
        let engines = fake_engines(Setup { chat_stream: Some((400, Duration::from_millis(10))), ..Setup::default() }).await;
        let (tunnel, _dir) = tunnel_with_engines("chat-cancelled", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x83, &chat_turn());
        let batch = json!({ "requests": [pending_row("chat-1", "chat", &row)] }).to_string();
        let frames = AtomicUsize::new(0);
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else if req.line() == "POST /api/v1/desktop-ai/chat-1/frames" && frames.fetch_add(1, Ordering::SeqCst) >= 3 {
                Reply::status(409, CANCELLED_BY_THE_SITE)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        let started = std::time::Instant::now();
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the turn slot", || tunnel.turn_slot.available_permits() == 1).await;
        assert!(started.elapsed() < Duration::from_secs(2), "stopped at its fourth frame, not at the end of the answer: {:?}", started.elapsed());
        // The engine stops writing once its reader has gone: the count of pieces written settles, well short of 400.
        tokio::time::sleep(Duration::from_millis(500)).await;
        let written = engines.chat_pieces.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(engines.chat_pieces.load(Ordering::SeqCst), written, "the answer is still being written");
        assert!(written < 200, "{written} of 400 pieces were written");
        assert_eq!(frames_posted(&server, "chat-1"), 4, "three frames taken, one answered cancelled, none after: {:?}", server.lines());
        assert!(completed_as(&server, "chat-1").is_none(), "no outcome is reported for it: {:?}", server.lines());
        assert!(tunnel.sessions.seal_outbound("chat-1", b"more").is_err(), "its session is gone");
        assert_eq!(engines.chats_asked().len(), 1);
    }

    /// Answered so to its last frame, a request has nothing left to stop: the picture was drawn and the answer
    /// written. It ends there all the same, with no outcome reported, both slots free and the lane untroubled.
    #[tokio::test]
    async fn a_request_the_site_cancelled_before_its_last_frame_ends_there() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        let engines = fake_engines(Setup::default()).await;
        let (tunnel, _dir) = tunnel_with_engines("final-cancelled", Some(engines.ui.clone()));
        let picture = sealed_to(&tunnel, 0x84, &sealed(json!({ "providerId": "oaiy-engine" })));
        let chat = sealed_to(&tunnel, 0x85, &chat_turn());
        let batch = json!({ "requests": [pending_row("picture-1", "image", &picture), pending_row("chat-1", "chat", &chat)] }).to_string();
        let polls = AtomicUsize::new(0);
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(if polls.fetch_add(1, Ordering::SeqCst) == 0 { &batch } else { r#"{"requests":[]}"# })
            } else if req.target.ends_with("/frames") {
                Reply::status(409, CANCELLED_BY_THE_SITE)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 2);
        until("both last frames", || frames_posted(&server, "picture-1") == 1 && frames_posted(&server, "chat-1") == 1).await;
        until("both slots", || tunnel.picture_slot.available_permits() == 1 && tunnel.turn_slot.available_permits() == 1).await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!((frames_posted(&server, "picture-1"), frames_posted(&server, "chat-1")), (1, 1), "{:?}", server.lines());
        assert!(!server.lines().iter().any(|l| l.ends_with("/complete")), "no outcome is reported: {:?}", server.lines());
        assert!(tunnel.sessions.seal_outbound("picture-1", b"more").is_err() && tunnel.sessions.seal_outbound("chat-1", b"more").is_err());
        // The work was done before the site said so: nothing was there to stop.
        assert_eq!((engines.pictures_asked().len(), engines.chats_asked().len(), engines.abandoned.load(Ordering::SeqCst)), (1, 1, 0));
        // And it is no trouble of the lane's: the next poll is an ordinary one.
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0);
    }

    /// The site answers the outcome of a request it has ended as it answers its frames. Ended between a request's
    /// last frame and its outcome, that is the same ordinary end; an outcome refused for another reason is still
    /// the lane's trouble to report.
    #[tokio::test]
    async fn an_outcome_the_site_answers_cancelled_is_an_ordinary_end_too() {
        use crate::link::testkit::{Provider, Reply};
        let (tunnel, dir) = tunnel_in("outcome-cancelled");
        let server = Provider::start(|req| {
            if req.target == "/api/v1/desktop-ai/ended/complete" {
                Reply::status(409, CANCELLED_BY_THE_SITE)
            } else {
                Reply::status(409, r#"{"error":true,"code":"not_claimed","message":"This request is not in a claimed state"}"#)
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());
        for (id, ordinary) in [("ended", true), ("other", false)] {
            // (a request that never opened and names no key: nothing can be sealed for it, so only its outcome is sent)
            let request = AiRequest { id: id.into(), kind: Some("chat".into()), provider_id: None, eph_pub: None, envelope: None };
            let lane = Lane { http: reqwest::Client::new(), account: &account, spec: &spec, instance: "oaiy-test", id };
            let reported = tunnel.finish(&lane, &request, Err(TunnelError::new("invalid_request", "the turn carries no chat key"))).await;
            assert_eq!(reported.is_ok(), ordinary, "{id}: {reported:?}");
        }
        assert_eq!(server.lines(), ["POST /api/v1/desktop-ai/ended/complete", "POST /api/v1/desktop-ai/other/complete"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Any other refusal of a frame is what it always was, a frame lost: a 409 that says anything else (or
    /// nothing), a 429, a 500. The picture is drawn to its end and the answer written to its end, and each is
    /// reported.
    #[tokio::test]
    async fn a_frame_refused_for_any_other_reason_is_a_frame_lost_and_the_work_goes_on() {
        use crate::link::testkit::{Provider, Reply};
        use std::sync::atomic::AtomicUsize;
        // The engines keep the picture until three of its progress frames have been refused.
        let refused = Arc::new(AtomicUsize::new(0));
        let hold = {
            let refused = refused.clone();
            move || refused.load(Ordering::SeqCst) >= 3
        };
        let engines = fake_engines(Setup { hold: Some(Arc::new(hold)), chat_stream: Some((6, Duration::from_millis(10))), ..Setup::default() }).await;
        let (mut tunnel, _dir) = tunnel_with_engines("frame-refused", Some(engines.ui.clone()));
        Arc::get_mut(&mut tunnel).unwrap().picture_progress = Duration::from_millis(100);
        let picture = sealed_to(&tunnel, 0x86, &sealed(json!({ "providerId": "oaiy-engine" })));
        let chat = sealed_to(&tunnel, 0x87, &chat_turn());
        let batch = json!({ "requests": [pending_row("picture-1", "image", &picture), pending_row("chat-1", "chat", &chat)] }).to_string();
        let (polls, chat_frames) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let server = {
            let refused = refused.clone();
            Provider::start(move |req| {
                if req.target.starts_with("/api/v1/desktop-ai/pending") {
                    Reply::ok(if polls.fetch_add(1, Ordering::SeqCst) == 0 { &batch } else { r#"{"requests":[]}"# })
                } else if req.line() == "POST /api/v1/desktop-ai/picture-1/frames" {
                    match refused.fetch_add(1, Ordering::SeqCst) {
                        0 => Reply::status(409, r#"{"error":true,"code":"lane_busy","message":"Another request holds the lane."}"#),
                        1 => Reply::status(409, "conflict"),
                        2 => Reply::status(500, r#"{"error":true,"code":"cancelled","message":"not a 409"}"#),
                        _ => Reply::ok("{}"),
                    }
                } else if req.line() == "POST /api/v1/desktop-ai/chat-1/frames" {
                    match chat_frames.fetch_add(1, Ordering::SeqCst) {
                        0 => Reply::status(409, r#"{"error":true,"code":"not_claimed"}"#),
                        1 => Reply::status(429, r#"{"error":true,"message":"API key rate limit exceeded. Try again in 3s."}"#),
                        _ => Reply::ok("{}"),
                    }
                } else {
                    Reply::ok("{}")
                }
            })
        };
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 2);
        until("the picture", || completed_as(&server, "picture-1").is_some()).await;
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        assert_eq!((completed_as(&server, "picture-1").as_deref(), completed_as(&server, "chat-1").as_deref()), (Some("done"), Some("done")));
        assert_eq!(frame_of(&tunnel, &server, "picture-1", 0x86)["picture"]["mimeType"], "image/png");
        // Every piece of the answer was written, and the turn went on to its last frame, its first two frames lost.
        assert_eq!(engines.chat_pieces.load(Ordering::SeqCst), 6, "{:?}", server.lines());
        assert!(frames_posted(&server, "chat-1") >= 2, "{:?}", server.lines());
        assert_eq!(frame_of(&tunnel, &server, "chat-1", 0x87), json!({ "v": 1, "type": "final", "kind": "final" }));
        assert_eq!(engines.abandoned.load(Ordering::SeqCst), 0, "the picture was not dropped");
    }

    /// The text a reader joins from a turn's delta frames: each frame's `choices[].delta.content`, as FormLogic reads.
    fn text_read(frames: &[Value]) -> String {
        frames.iter().map(|f| f["delta"]["choices"][0]["delta"]["content"].as_str().unwrap_or_default()).collect()
    }

    /// A long answer goes in a few frames a second, and none of it is lost. A frame a chunk, as it was, is a
    /// request to the site for each token a model writes: 282 in five seconds of a real answer, against the 600 a
    /// minute a link's key may make, so that one long answer had every other request of its link refused.
    #[tokio::test]
    async fn a_long_streamed_answer_goes_in_a_few_frames_a_second_and_loses_no_text() {
        use crate::link::testkit::{Provider, Reply};
        // 300 chunks, 10 ms and more apart: a model writing fast for three seconds or longer.
        const PIECES: usize = 300;
        let engines = fake_engines(Setup { chat_stream: Some((PIECES, Duration::from_millis(10))), ..Setup::default() }).await;
        let (tunnel, _dir) = tunnel_with_engines("chat-gathered", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x88, &chat_turn());
        let batch = json!({ "requests": [pending_row("chat-1", "chat", &row)] }).to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        assert_eq!(completed_as(&server, "chat-1").as_deref(), Some("done"));
        assert_eq!(engines.chat_pieces.load(Ordering::SeqCst), PIECES);

        let mut frames = frames_of(&tunnel, &server, "chat-1", 0x88);
        assert_eq!(frames.pop().unwrap(), json!({ "v": 1, "type": "final", "kind": "final" }), "the answer was streamed: its last frame carries no text");
        assert!(frames.iter().all(|f| f["kind"] == "delta" && f["type"] == "delta"), "{frames:?}");
        // None of it is lost, and it reads in order: what a reader joins from the frames is what the model wrote.
        let written: String = (0..PIECES).map(|i| format!("word{i} ")).collect();
        assert_eq!(text_read(&frames), written);
        // The first frame went at once with the start of the answer (its first word or three, not an interval's
        // worth of them). With it went what came before the first word: who speaks, and the newest word of the
        // prompt's progress, which had been two chunks with no choices.
        let first = &frames[0]["delta"]["choices"][0]["delta"];
        let began = first["content"].as_str().unwrap();
        assert!(began.starts_with("word0 ") && began.len() <= "word0 word1 word2 ".len(), "{first}");
        assert_eq!(first["role"], "assistant");
        assert_eq!(frames[0]["delta"]["oaiy_progress"], json!({ "prompt_done": 3, "prompt_total": 3 }), "{}", frames[0]);
        // The end says why the model stopped and what it used, as the provider's last chunks did: its last frame
        // carries what it used, and why it stopped is in that frame or, where an interval ended between the
        // provider's last two chunks (as it does on a slow machine), in the frame before it. Neither is lost.
        let last = &frames.last().unwrap()["delta"];
        assert_eq!(last["usage"]["completion_tokens"], json!(PIECES), "{last}");
        let stopped: Vec<(usize, &str)> = frames.iter().enumerate().filter_map(|(at, f)| f["delta"]["choices"][0]["finish_reason"].as_str().map(|why| (at, why))).collect();
        assert_eq!(stopped.len(), 1, "why it stopped is said once: {stopped:?}");
        assert_eq!(stopped[0].1, "stop");
        assert!(stopped[0].0 + 2 >= frames.len(), "and at the end: frame {} of {}", stopped[0].0 + 1, frames.len());
        assert_eq!((&last["id"], &last["model"]), (&json!("cmpl-1"), &json!("flash")));

        // A few frames a second, where there was one a chunk. While the answer is written its frames are an interval
        // apart, so never more than five in any second; when it ends, the last of its text and the final frame
        // follow at once (two frames, once a turn).
        let seen = server.requests();
        let posted: Vec<std::time::Instant> = seen.iter().filter(|r| r.line() == "POST /api/v1/desktop-ai/chat-1/frames").map(|r| r.at).collect();
        assert_eq!(posted.len(), frames.len() + 1);
        let while_written = &posted[..posted.len() - 2];
        for pair in while_written.windows(2) {
            // (as the site saw them arrive, so a few milliseconds are allowed for)
            assert!(pair[1] - pair[0] >= DELTA_FLUSH_INTERVAL - Duration::from_millis(10), "two frames {:?} apart", pair[1] - pair[0]);
        }
        for six in while_written.windows(6) {
            assert!(six[5] - six[0] >= Duration::from_secs(1), "six frames in {:?} ({} frames in all)", six[5] - six[0], posted.len());
        }
        let took = posted[posted.len() - 1] - posted[0];
        assert!(posted.len() as f64 <= 5.0 * took.as_secs_f64() + 3.0, "{} frames in {took:?}", posted.len());
        assert!(frames.len() < PIECES / 5, "{} frames for {PIECES} chunks", frames.len());
        // And it streamed: the first frame came promptly, long before the last, with frames between.
        let claimed = seen.iter().find(|r| r.line() == "POST /api/v1/desktop-ai/chat-1/claim").unwrap().at;
        assert!(posted[0] - claimed < Duration::from_secs(2), "the first frame waited {:?}", posted[0] - claimed);
        assert!(took >= Duration::from_secs(1) && frames.len() >= 4, "{} frames in {took:?}", frames.len());
    }

    /// Before the answer begins there is only word of the turn (who speaks, how far the prompt has got). It does not
    /// go a frame an interval, nobody reading it as it comes, but once a second while the first word is awaited, so
    /// that the site hears of a turn whose prompt is long in going in; and the first word then goes at once.
    #[tokio::test]
    async fn word_of_a_turn_whose_answer_has_not_begun_goes_once_a_second() {
        use crate::link::testkit::{Provider, Reply};
        // A chunk every 300 ms: who speaks, the prompt's progress twice, then the first word 1.2 s in.
        let engines = fake_engines(Setup { chat_stream: Some((4, Duration::from_millis(300))), ..Setup::default() }).await;
        let (tunnel, _dir) = tunnel_with_engines("chat-slow-start", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x8a, &chat_turn());
        let batch = json!({ "requests": [pending_row("chat-1", "chat", &row)] }).to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        let mut frames = frames_of(&tunnel, &server, "chat-1", 0x8a);
        assert_eq!(frames.pop().unwrap(), json!({ "v": 1, "type": "final", "kind": "final" }));
        assert_eq!(text_read(&frames), "word0 word1 word2 word3 ");
        // One frame before the first word, about a second in and not at each of the three chunks it was made of.
        let words_from = frames.iter().position(|f| !text_read(std::slice::from_ref(f)).is_empty()).unwrap();
        assert_eq!(words_from, 1, "{frames:?}");
        let before = &frames[0]["delta"];
        assert_eq!((&before["choices"][0]["delta"]["role"], &before["oaiy_progress"]["prompt_total"]), (&json!("assistant"), &json!(3)), "{before}");
        assert!(frames[..2].iter().any(|f| f["delta"]["oaiy_progress"]["prompt_done"] == 3), "the newest word of the prompt went on: {frames:?}");
        let seen = server.requests();
        let posted: Vec<std::time::Instant> = seen.iter().filter(|r| r.line() == "POST /api/v1/desktop-ai/chat-1/frames").map(|r| r.at).collect();
        let claimed = seen.iter().find(|r| r.line() == "POST /api/v1/desktop-ai/chat-1/claim").unwrap().at;
        assert!(posted[0] - claimed >= Duration::from_millis(900), "it went {:?} after the claim", posted[0] - claimed);
        // The first word did not wait for an interval after it.
        assert_eq!(frames[1]["delta"]["choices"][0]["delta"]["content"], "word0 ");
    }

    /// A chunk that is not text to join to other text (a tool call's piece) goes as it came, in its place: what was
    /// waiting goes before it and what follows after, so a reader gets everything in the provider's order.
    #[tokio::test]
    async fn a_chunk_that_cannot_be_joined_goes_as_it_came_in_its_place() {
        use crate::link::testkit::{Provider, Reply};
        let odd = json!({ "id": "cmpl-1", "choices": [{ "index": 0, "delta": { "tool_calls": [{ "index": 0, "function": { "name": "look_up", "arguments": "{\"q\":" } }] } }] });
        let engines = fake_engines(Setup { chat_stream: Some((12, Duration::from_millis(10))), chat_stream_odd: Some((5, odd.clone())), ..Setup::default() }).await;
        let (tunnel, _dir) = tunnel_with_engines("chat-odd-chunk", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x89, &chat_turn());
        let batch = json!({ "requests": [pending_row("chat-1", "chat", &row)] }).to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the chat turn", || completed_as(&server, "chat-1").is_some()).await;
        let mut frames = frames_of(&tunnel, &server, "chat-1", 0x89);
        assert_eq!(frames.pop().unwrap(), json!({ "v": 1, "type": "final", "kind": "final" }));
        let at = frames.iter().position(|f| f["delta"] == odd).unwrap_or_else(|| panic!("the odd chunk is not among the frames as it came: {frames:?}"));
        let words = |range: std::ops::Range<usize>| range.map(|i| format!("word{i} ")).collect::<String>();
        assert_eq!(text_read(&frames[..at]), words(0..5), "what came before it is sent before it: {frames:?}");
        assert_eq!(text_read(&frames[at + 1..]), words(5..12), "and what came after it, after: {frames:?}");
    }

    /// Chunks gathered into one read the same as the chunks did: the text joined in order, the newest of everything
    /// else. What cannot be joined so is left for the caller to send as it came.
    #[test]
    fn chunks_of_a_stream_are_gathered_into_one_that_reads_the_same() {
        let chunk = |delta: Value, finish: Value| {
            json!({ "id": "c1", "object": "chat.completion.chunk", "created": 7, "model": "m", "choices": [{ "index": 0, "delta": delta, "finish_reason": finish, "logprobs": null }] })
        };
        let mut gathered = Gathered::default();
        assert!(gathered.is_empty() && !gathered.has_text());
        // One chunk in, the same chunk out.
        let only = chunk(json!({ "role": "assistant", "content": "" }), Value::Null);
        assert!(gathered.take(&only));
        assert!(!gathered.is_empty() && !gathered.has_text(), "a role is not text");
        assert_eq!(gathered.chunk(), only);
        assert!(gathered.is_empty());

        // Several: the answer's text joined in order and the model's reasoning beside it, the newest of the rest,
        // and a null never in a value's place.
        for piece in [
            json!({ "role": "assistant", "content": null }),
            json!({ "reasoning_content": "Let me " }),
            json!({ "reasoning_content": "think." }),
            json!({ "content": "Hel" }),
            json!({ "content": "lo, " }),
            json!({ "content": "wörld" }),
        ] {
            assert!(gathered.take(&chunk(piece, Value::Null)));
        }
        let mut last = chunk(json!({}), json!("length"));
        last["usage"] = json!({ "completion_tokens": 5 });
        assert!(gathered.take(&last));
        assert!(gathered.has_text() && !gathered.is_full());
        assert_eq!(
            gathered.chunk(),
            json!({
                "id": "c1", "object": "chat.completion.chunk", "created": 7, "model": "m", "usage": { "completion_tokens": 5 },
                "choices": [{ "index": 0, "finish_reason": "length", "logprobs": null,
                    "delta": { "role": "assistant", "content": "Hello, wörld", "reasoning_content": "Let me think." } }],
            })
        );

        // A chunk with no choices is word of how the turn is getting on, of which the newest goes on: by itself
        // while the prompt goes in (still with no choices), or beside the text once there is some.
        let progress = |done: u64| json!({ "id": "c1", "choices": [], "oaiy_progress": { "prompt_done": done, "prompt_total": 40 } });
        assert!(gathered.take(&progress(0)) && gathered.take(&progress(24)));
        assert!(!gathered.is_empty() && !gathered.has_text());
        assert_eq!(gathered.chunk(), progress(24));
        assert!(gathered.take(&progress(40)) && gathered.take(&chunk(json!({ "content": "Hi" }), Value::Null)));
        assert!(gathered.take(&json!({ "id": "c1", "choices": [], "usage": { "total_tokens": 9 } })));
        let with_text = gathered.chunk();
        assert_eq!(with_text["choices"][0]["delta"], json!({ "content": "Hi" }));
        assert_eq!((&with_text["oaiy_progress"]["prompt_done"], &with_text["usage"]["total_tokens"]), (&json!(40), &json!(9)), "{with_text}");

        // Not to be joined: no chunk at all, more or other choices than the first, a tool call's piece, per-token
        // detail, a whole message, and an object with no `choices` (an error, a side note of a provider's own).
        for odd in [
            json!("text"),
            json!({ "oaiy_reasoning_preview": "weighing the two" }),
            json!({ "choices": [{ "index": 0, "delta": { "content": "a" } }, { "index": 1, "delta": { "content": "b" } }] }),
            json!({ "choices": [{ "index": 1, "delta": { "content": "the second choice" } }] }),
            json!({ "choices": [{ "index": 0, "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "{\"a\"" } }] } }] }),
            json!({ "choices": [{ "index": 0, "delta": { "content": "a" }, "logprobs": { "content": [] } }] }),
            json!({ "choices": [{ "index": 0, "message": { "content": "whole" } }] }),
            json!({ "error": { "message": "overloaded" } }),
        ] {
            assert!(!gathered.take(&odd), "{odd}");
            assert!(gathered.is_empty(), "{odd}");
        }

        // More than a frame ought to carry does not wait for its interval.
        assert!(gathered.take(&chunk(json!({ "content": "x".repeat(GATHER_AT_MOST_BYTES) }), Value::Null)));
        assert!(gathered.is_full());
    }

    /// Contract §0.13: the answer's limit a chat request names (`maxTokens`) reaches the model as `max_tokens`, in
    /// a streamed request as in a whole one. A missing or silly one sends none, and the provider's own applies.
    #[tokio::test]
    async fn a_chat_requests_limit_on_its_answer_reaches_the_engine_and_a_silly_one_does_not() {
        let engines = fake_engines(Setup::default()).await;
        let (tunnel, dir) = tunnel_with_engines("answer-limit", Some(engines.ui.clone()));
        // The engine answers in one piece, so nothing is posted to the account on this path.
        let account = linked("http://127.0.0.1:9".into(), "flk_ai");
        let spec = descriptor::builtin().remove(0).desktop_ai.unwrap();
        let lane = Lane { http: reqwest::Client::new(), account: &account, spec: &spec, instance: "oaiy-test", id: "turn-1" };
        let cases = [
            (Some(json!(1500)), Some(1500)),
            (Some(json!(1)), Some(1)),
            (Some(json!(MAX_ANSWER_TOKENS)), Some(MAX_ANSWER_TOKENS)),
            (None, None),
            (Some(Value::Null), None),
            (Some(json!(0)), None),
            (Some(json!(-200)), None),
            (Some(json!(1500.5)), None),
            (Some(json!("1500")), None),
            (Some(json!(true)), None),
            (Some(json!([1500])), None),
            (Some(json!(MAX_ANSWER_TOKENS + 1)), None),
        ];
        for (asked, reaches) in cases {
            let mut body = json!({ "v": 1, "messages": [{ "role": "user", "content": "hi" }] });
            if let Some(n) = &asked {
                body["maxTokens"] = n.clone();
            }
            tunnel.chat(&lane, "oaiy-engine", None, &body, "unused").await.unwrap();
            let sent = engines.chats_asked().pop().unwrap().body;
            assert_eq!(sent.get("max_tokens").and_then(Value::as_u64), reaches, "maxTokens {asked:?}: {sent}");
            assert_eq!(sent.get("max_tokens").is_some(), reaches.is_some(), "maxTokens {asked:?}: {sent}");
            assert!(sent.get("maxTokens").is_none(), "the lane's own name for it is not the model's: {sent}");
            assert_eq!(sent["stream"], true, "asked as a stream, as every turn of the engine is: {sent}");
        }
        assert_eq!(answer_limit(&json!({ "maxTokens": 64 })), Some(64));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A picture whose claim another computer won is not made: the engines are never asked, and the slot is free again.
    #[tokio::test]
    async fn a_picture_whose_claim_is_lost_is_never_made() {
        use crate::link::testkit::{Provider, Reply};
        let engines = fake_engines(Setup::default()).await;
        let (tunnel, _dir) = tunnel_with_engines("picture-lost", Some(engines.ui.clone()));
        let row = sealed_to(&tunnel, 0x44, &sealed(json!({})));
        let batch = json!({ "requests": [pending_row("picture-1", "image", &row)] }).to_string();
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(&batch)
            } else if req.target.ends_with("/claim") {
                Reply::status(409, r#"{"code":"already_claimed"}"#)
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 0, "a lost claim is not work done");
        // (time for anything that had been started to reach the engines)
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(engines.pictures_asked().is_empty(), "{:?}", engines.pictures_asked());
        assert_eq!(tunnel.picture_slot.available_permits(), 1, "the slot is free again");
        assert!(!server.lines().iter().any(|l| l.ends_with("/frames") || l.ends_with("/complete")), "{:?}", server.lines());
    }

    /// A picture request that does not open, or that the engines refuse, is answered in the contract's shape:
    /// `{error: {code, message}}`, with a code of its own list.
    #[tokio::test]
    async fn a_refused_picture_is_answered_in_the_contracts_shape() {
        use crate::link::testkit::{Provider, Reply};
        let engines = fake_engines(Setup::default()).await;
        let (tunnel, _dir) = tunnel_with_engines("picture-refused", Some(engines.ui.clone()));
        // Sealed with another key than the row names: it does not open, and the error goes back under the row's.
        let (garbled_key, _) = sealed_to(&tunnel, 0x55, &json!({}));
        let (_, other) = sealed_to(&tunnel, 0x56, &sealed(json!({})));
        let disabled = sealed_to(&tunnel, 0x57, &sealed(json!({ "model": "qwen-image-2.1" })));
        let batch = json!({ "requests": [pending_row("garbled", "image", &(garbled_key, other))] }).to_string();
        let later = json!({ "requests": [pending_row("disabled", "image", &disabled)] }).to_string();
        let polls = std::sync::atomic::AtomicUsize::new(0);
        let server = Provider::start(move |req| {
            if req.target.starts_with("/api/v1/desktop-ai/pending") {
                Reply::ok(if polls.fetch_add(1, Ordering::SeqCst) == 0 { &batch } else { &later })
            } else {
                Reply::ok("{}")
            }
        });
        let (account, spec) = (linked(server.base.clone(), "flk_ai"), descriptor::builtin().remove(0).desktop_ai.unwrap());

        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the garbled picture", || completed_as(&server, "garbled").is_some()).await;
        assert_eq!(completed_as(&server, "garbled").as_deref(), Some("failed"));
        let refused = frame_of(&tunnel, &server, "garbled", 0x55);
        assert_eq!(refused["error"]["code"], "bad_request", "{refused}");
        assert_eq!(refused["kind"], "final", "an error ends a picture as its picture would: {refused}");

        // The site has the first answer before its task lets go of the slot, and a poll made while the slot is
        // held claims no picture: wait for it, as the other tests here do before they ask for the next one.
        until("the picture slot", || tunnel.picture_slot.available_permits() == 1).await;
        assert_eq!(tunnel.poll_cycle(&account, &spec, "oaiy-test").await.unwrap(), 1);
        until("the disabled picture", || completed_as(&server, "disabled").is_some()).await;
        let refused = frame_of(&tunnel, &server, "disabled", 0x57);
        assert_eq!(refused["error"]["code"], "model_disabled", "{refused}");
        assert!(refused["error"]["message"].as_str().unwrap().contains("qwen-image-2.1 is disabled"), "{refused}");
        assert_eq!(engines.pictures_asked().len(), 1, "the engines said why; no picture was made");
    }

    /// The site lists the engine's picture models through the lane, each with its type, on a computer that has no
    /// language model chosen: listing needs none, though a chat turn still does.
    #[tokio::test]
    async fn the_engines_picture_models_are_listed_with_no_language_model_chosen() {
        let engines = fake_engines(Setup { llm: String::new(), ..Setup::default() }).await;
        let (tunnel, _dir) = tunnel_with_engines("picture-models", Some(engines.ui.clone()));
        let listed: Value = serde_json::from_slice(&tunnel.list_models("oaiy-engine").await.unwrap()).unwrap();
        assert_eq!(listed["models"][0], json!({ "id": "flux-3-klein", "type": "image" }), "{listed}");
        let Err(chat) = tunnel.resolve("oaiy-engine").await else { panic!("a chat turn needs a language model chosen") };
        assert!(chat.message.contains("no language model chosen"), "{}", chat.message);
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
