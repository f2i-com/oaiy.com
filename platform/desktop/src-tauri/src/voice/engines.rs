//! Speech for calls: speech-to-text and text-to-speech from OAIY's own voice
//! server (`oaiy-voice`: Parakeet and Qwen3-TTS in Rust on the GPU, in the
//! voice chosen from the clips in `<data>/voices`), run as the `oaiy-voice`
//! service and started when a call needs it. Aokie's servers answer the same
//! routes, and remain as the `aokie-stt` / `aokie-tts` services.

use std::time::{Duration, Instant};

use base64::Engine as _;
use futures_util::StreamExt;
use tokio::sync::mpsc;

use super::audio::{self, Resampler};
use crate::services::registry::RegistryHandle;

/// One process serves both: speech-to-text and text-to-speech.
pub const STT_SERVICE: &str = "oaiy-voice";
pub const TTS_SERVICE: &str = "oaiy-voice";
/// The rate calls run at on the wire (Aokie's realtime stream).
pub const WIRE_RATE: u32 = 24_000;
/// What speech-to-text takes.
pub const STT_RATE: u32 = 16_000;
/// How long a starting service may take to answer (it loads its model first).
const START_WAIT: Duration = Duration::from_secs(180);
/// How long a worker the heartbeat found not answering is given to answer when asked once more.
const ASKED_AGAIN: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct Engines {
    registry: Option<RegistryHandle>,
    http: reqwest::Client,
    /// Fixed addresses (tests): no services are started.
    fixed: Option<(String, String)>,
    /// The voices copied to a worker whose speech is used in place of this computer's (OAIY Network), by worker,
    /// voice and clip: the name each is kept under there.
    copied: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
    /// A worker's speech used whatever the network's settings say (tests).
    pinned: Option<crate::network::ServiceTarget>,
    /// A call's own ([`Engines::for_call`]): where its speech was first heard, kept for the rest of the call.
    call: Option<std::sync::Arc<tokio::sync::Mutex<Option<Chosen>>>>,
}

/// Where a call's speech is heard and spoken, chosen as the call begins: a worker's, as it was reached then, or this
/// computer's own.
#[derive(Clone)]
enum Chosen {
    There(crate::network::ServiceTarget),
    Here,
}

/// Where a speech service answers: the client that reaches it and its address, this computer's own or a paired
/// worker's chosen in its place (`remote`).
struct Target {
    http: reqwest::Client,
    base: String,
    remote: Option<crate::network::ServiceTarget>,
}

impl Target {
    /// A worker's.
    fn on(remote: crate::network::ServiceTarget) -> Self {
        Self { http: remote.client.clone(), base: remote.base.clone(), remote: Some(remote) }
    }
}

impl Engines {
    pub fn new(registry: RegistryHandle) -> Self {
        Self { registry: Some(registry), http: reqwest::Client::new(), fixed: None, copied: Default::default(), pinned: None, call: None }
    }

    /// Speech on a paired worker, whatever the network's settings say (tests).
    pub fn via(remote: crate::network::ServiceTarget) -> Self {
        Self { registry: None, http: reqwest::Client::new(), fixed: None, copied: Default::default(), pinned: Some(remote), call: None }
    }

    /// Speech on a paired worker, this computer's own (`registry`'s) answering where the worker does not (tests).
    #[cfg(test)]
    pub(crate) fn via_or_here(remote: crate::network::ServiceTarget, registry: RegistryHandle) -> Self {
        Self { registry: Some(registry), http: reqwest::Client::new(), fixed: None, copied: Default::default(), pinned: Some(remote), call: None }
    }

    /// Speech at fixed addresses (tests, or a voice server run by hand).
    pub fn at(stt: &str, tts: &str) -> Self {
        Self { registry: None, http: reqwest::Client::new(), fixed: Some((stt.to_string(), tts.to_string())), copied: Default::default(), pinned: None, call: None }
    }

    /// The same speech for one call: where its speech is first heard (as the call begins) is kept for the rest of it,
    /// and each of its lines goes there. A line went wherever the network said then: one heartbeat missed sent the next
    /// to this computer's own OAIY Voice, which could be starting cold for minutes in the middle of the call, and the
    /// heartbeat after it sent the lines back, in another voice.
    pub fn for_call(&self) -> Self {
        Self { call: Some(Default::default()), ..self.clone() }
    }

    /// Where a speech service answers now: a paired worker's chosen in place of this computer's (OAIY Network, Services
    /// → Other computers), else this computer's own, started if it is not running; either waited on until it answers.
    /// That computer not answering (asked now: once more, briefly, where the heartbeat found it not answering), this
    /// computer's own OAIY Voice answers where it is set up here, said in the log. A call's is chosen once
    /// ([`Engines::for_call`]): its other lines go where its first did, and that worker no longer answering in the
    /// middle of it, they go to this computer's own only where that runs already (not started then, with the caller
    /// waiting), and stay there; elsewhere the line is not heard, and the worker is asked again at the next.
    async fn target(&self, id: &'static str) -> Result<Target, String> {
        let Some(call) = &self.call else { return self.choose(id).await.map(|(t, _)| t) };
        let mut chosen = call.lock().await;
        let remote = match chosen.clone() {
            None => {
                let (target, choice) = self.choose(id).await?;
                *chosen = Some(choice);
                return Ok(target);
            }
            Some(Chosen::Here) => {
                drop(chosen);
                return Ok(Target { http: self.http.clone(), base: self.base(id).await?, remote: None });
            }
            Some(Chosen::There(remote)) => remote,
        };
        drop(chosen);
        // (the heartbeat's word on it now: one found not answering is asked once, briefly)
        let offline = self.remote_now(id).is_some_and(|now| now.fingerprint == remote.fingerprint && now.offline);
        match self.reach(id, &remote, offline).await {
            Ok(()) => Ok(Target::on(remote)),
            Err((e, true)) => {
                let Some(base) = self.running_here(id).await else { return Err(e) };
                *call.lock().await = Some(Chosen::Here);
                log::warn!("voice: {e}; this computer's own OAIY Voice, running already, answers for the rest of the call");
                Ok(Target { http: self.http.clone(), base, remote: None })
            }
            Err((e, false)) => Err(e),
        }
    }

    /// Where a speech service answers, chosen now ([`Engines::target`]), and what was chosen.
    async fn choose(&self, id: &'static str) -> Result<(Target, Chosen), String> {
        let Some(remote) = self.remote_now(id) else {
            return Ok((Target { http: self.http.clone(), base: self.base(id).await?, remote: None }, Chosen::Here));
        };
        let unanswered = match self.reach(id, &remote, remote.offline).await {
            Ok(()) => return Ok((Target::on(remote.clone()), Chosen::There(remote))),
            Err((e, true)) if self.installed_here(id) => e,
            Err((e, _)) => return Err(e),
        };
        // (said once a minute at most: a call asks for each line)
        {
            static SAID: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
            let mut said = SAID.lock().unwrap_or_else(|p| p.into_inner());
            if said.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                *said = Some(Instant::now());
                log::warn!("voice: {unanswered}; this computer's own OAIY Voice answers meanwhile");
            }
        }
        Ok((Target { http: self.http.clone(), base: self.base(id).await?, remote: None }, Chosen::Here))
    }

    /// The worker whose service `id` is used in place of this computer's, as the network says now (or as pinned).
    fn remote_now(&self, id: &'static str) -> Option<crate::network::ServiceTarget> {
        match &self.pinned {
            Some(r) => Some(r.clone()),
            None if self.fixed.is_none() && self.registry.is_some() => crate::network::service_target(id),
            None => None,
        }
    }

    /// This computer's own service `id`'s address where it runs and answers now, nothing started.
    async fn running_here(&self, id: &'static str) -> Option<String> {
        let port = self.registry.as_ref()?.lock().ok()?.running_service_lease(id)?.port;
        let base = format!("http://127.0.0.1:{port}");
        let r = self.http.get(format!("{base}/v1/models")).timeout(ASKED_AGAIN).send().await.ok()?;
        r.status().is_success().then_some(base)
    }

    /// A worker's speech service, asked until it answers (it starts there when asked): why not, and whether that is
    /// the worker itself not answering (`true`), not its service. `briefly`: the heartbeat found it not answering, and
    /// it is asked once, for a moment ([`ASKED_AGAIN`]).
    async fn reach(&self, id: &'static str, remote: &crate::network::ServiceTarget, briefly: bool) -> Result<(), (String, bool)> {
        let started = Instant::now();
        let mut unreachable = 0;
        let (tries, wait) = if briefly { (1, ASKED_AGAIN) } else { (2, Duration::from_secs(5)) };
        loop {
            match remote.client.get(format!("{}/v1/models", remote.base)).timeout(wait).send().await {
                Ok(r) if r.status().is_success() => return Ok(()),
                Ok(r) if !matches!(r.status().as_u16(), 503 | 502) => {
                    return Err((format!("{id} on {}: HTTP {}: {}", remote.node_name, r.status(), r.text().await.unwrap_or_default()), false));
                }
                // (the worker itself not answering, not its service starting: said at once, not waited on for each
                // line of a call; asked a second time for a moment's dropout, where the heartbeat heard it last)
                Err(e) => {
                    unreachable += 1;
                    if unreachable >= tries {
                        return Err((format!("{} does not answer ({id} is used there): {e}", remote.node_name), true));
                    }
                    tokio::time::sleep(Duration::from_millis(700)).await;
                }
                _ if started.elapsed() > START_WAIT => return Err((format!("{id} on {} did not answer within {} s", remote.node_name, START_WAIT.as_secs()), false)),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }

    /// Whether this computer has its own of service `id` set up, to answer in a worker's place.
    fn installed_here(&self, id: &str) -> bool {
        self.registry.as_ref().and_then(|r| r.lock().ok()).is_some_and(|r| r.snapshot().services.iter().any(|s| s.id == id && s.installed))
    }

    /// The voice as `target` knows it: this computer's own name, or on a worker the copy of its clip kept there (copied
    /// the first time it is used there, and again when its clip changes). A voice that cannot be copied is asked for
    /// by its own name (the worker may have it).
    async fn voice_at(&self, target: &Target, voice: &str) -> String {
        let Some(remote) = &target.remote else { return voice.to_string() };
        let Some(clip) = super::voices::list().into_iter().find(|c| c.name == voice) else { return voice.to_string() };
        let Some(dir) = super::voices::dir() else { return voice.to_string() };
        let path = dir.join(&clip.file);
        // (copied again when its clip or its words change, or to another computer at that address)
        let changed = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis());
        let key = format!("{}|{}|{}|{}|{}|{}", remote.fingerprint, clip.name, clip.file, clip.bytes, changed(&path), changed(&path.with_extension("txt")));
        if let Some(name) = self.copied.lock().ok().and_then(|c| c.get(&key).cloned()) {
            return name;
        }
        let words = std::fs::read_to_string(path.with_extension("txt")).ok();
        let Ok(bytes) = tokio::fs::read(&path).await else { return voice.to_string() };
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("wav").to_string();
        let body = serde_json::json!({ "clip": base64::engine::general_purpose::STANDARD.encode(bytes), "extension": extension, "words": words });
        let name = percent_encoding::utf8_percent_encode(&clip.name, percent_encoding::NON_ALPHANUMERIC);
        let answer = remote.client.put(format!("{}/worker/v1/voices/{name}", remote.root)).json(&body).timeout(Duration::from_secs(60)).send().await;
        match answer {
            Ok(r) if r.status().is_success() => {
                let name = r.json::<serde_json::Value>().await.ok().and_then(|v| v["voice"].as_str().map(str::to_owned)).unwrap_or_else(|| voice.to_string());
                if let Ok(mut c) = self.copied.lock() {
                    c.insert(key, name.clone());
                }
                name
            }
            Ok(r) => {
                log::warn!("voice: {voice} could not be copied to {}: HTTP {}", remote.node_name, r.status());
                voice.to_string()
            }
            Err(e) => {
                log::warn!("voice: {voice} could not be copied to {}: {e}", remote.node_name);
                voice.to_string()
            }
        }
    }

    /// A worker that did not know a voice copied there (it was removed, or the worker set up again): copied again
    /// the next time it is spoken in.
    fn forget_copies(&self, target: &Target, status: reqwest::StatusCode) {
        if target.remote.is_some() && matches!(status.as_u16(), 400 | 404) {
            if let Ok(mut c) = self.copied.lock() {
                c.clear();
            }
        }
    }

    /// A speech service's address on this computer, started if it is not running and waited on until it answers.
    async fn base(&self, id: &'static str) -> Result<String, String> {
        if let Some((stt, tts)) = &self.fixed {
            return Ok(if id == STT_SERVICE { stt.clone() } else { tts.clone() });
        }
        let registry = self.registry.clone().ok_or("no services")?;
        let port = tokio::task::spawn_blocking(move || -> Result<u16, String> {
            let mut r = registry.lock().map_err(|_| "the services are unavailable".to_string())?;
            r.start(id).map_err(|e| format!("{id} did not start: {e}"))?;
            r.service_port(id).ok_or_else(|| format!("{id} has no port"))
        })
        .await
        .map_err(|e| e.to_string())??;
        let base = format!("http://127.0.0.1:{port}");
        let started = Instant::now();
        loop {
            match self.http.get(format!("{base}/v1/models")).timeout(Duration::from_secs(3)).send().await {
                Ok(r) if r.status().is_success() => return Ok(base),
                _ if started.elapsed() > START_WAIT => return Err(format!("{id} did not answer within {} s", START_WAIT.as_secs())),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }

    /// Start both (a call is coming: the first words should not wait for a model to load).
    pub async fn warm(&self) -> Result<(), String> {
        if STT_SERVICE == TTS_SERVICE && self.fixed.is_none() {
            return self.target(STT_SERVICE).await.map(|_| ());
        }
        let (a, b) = tokio::join!(self.target(STT_SERVICE), self.target(TTS_SERVICE));
        a.and(b).map(|_| ())
    }

    /// What was said in an utterance (24 kHz phone audio).
    pub async fn transcribe(&self, utterance: &[i16]) -> Result<String, String> {
        let pcm = Resampler::new(WIRE_RATE, STT_RATE).process(utterance);
        self.transcribe_wav(audio::wav(&pcm, STT_RATE)).await
    }

    /// What was said in a recording: a 16 kHz mono 16-bit WAV.
    pub async fn transcribe_wav(&self, wav: Vec<u8>) -> Result<String, String> {
        let t = self.target(STT_SERVICE).await?;
        let body = serde_json::json!({
            "audio": base64::engine::general_purpose::STANDARD.encode(wav),
            "response_format": "json",
        });
        let resp = t.http.post(format!("{}/v1/audio/transcriptions", t.base)).json(&body).timeout(Duration::from_secs(60)).send().await.map_err(|e| format!("speech-to-text: {e}"))?;
        let status = resp.status();
        // (the status first: a failure's answer is often not JSON, and reading it as JSON hid the HTTP error behind a
        // parse error)
        if !status.is_success() {
            let said = resp.text().await.unwrap_or_default();
            return Err(format!("speech-to-text: HTTP {status}: {}", said.chars().take(400).collect::<String>()));
        }
        let value: serde_json::Value = resp.json().await.map_err(|e| format!("speech-to-text answer: {e}"))?;
        Ok(value.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string())
    }

    /// Have `text` made whole and kept by the speech server, to be said from memory when it is asked for: a caller's
    /// greeting, while the line settles. The server makes it again with another seed where it does not sound like its
    /// voice (one line in twenty or so does not, and a greeting so made was the same at every call).
    /// Answers with what the server said of it (`prepared`, and where it was made, its `likeness` and `seed`).
    pub async fn prepare(&self, text: &str, voice: Option<&str>) -> Result<serde_json::Value, String> {
        let t = self.target(TTS_SERVICE).await?;
        let mut body = serde_json::json!({ "input": text, "response_format": "pcm", "prepare": true });
        if let Some(v) = voice {
            body["voice"] = serde_json::Value::String(self.voice_at(&t, v).await);
        }
        let resp = t.http.post(format!("{}/v1/audio/speech", t.base)).json(&body).timeout(Duration::from_secs(60)).send().await.map_err(|e| format!("text-to-speech: {e}"))?;
        if !resp.status().is_success() {
            self.forget_copies(&t, resp.status());
            return Err(format!("text-to-speech: HTTP {}", resp.status()));
        }
        Ok(resp.json().await.unwrap_or(serde_json::Value::Null))
    }

    /// Speak `text`: its audio at the wire rate, chunk by chunk as it is made,
    /// until done or `out` is closed (the item was cancelled).
    pub async fn speak(&self, text: &str, voice: Option<&str>, out: &mpsc::Sender<Vec<i16>>) -> Result<(), String> {
        let t = self.target(TTS_SERVICE).await?;
        let mut body = serde_json::json!({ "input": text, "response_format": "pcm" });
        if let Some(v) = voice {
            body["voice"] = serde_json::Value::String(self.voice_at(&t, v).await);
        }
        let resp = t.http.post(format!("{}/v1/audio/speech", t.base)).json(&body).timeout(Duration::from_secs(120)).send().await.map_err(|e| format!("text-to-speech: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            self.forget_copies(&t, status);
            return Err(format!("text-to-speech: HTTP {status}: {}", resp.text().await.unwrap_or_default()));
        }
        let rate = resp.headers().get("x-sample-rate").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u32>().ok()).unwrap_or(WIRE_RATE);
        let mut resample = Resampler::new(rate, WIRE_RATE);
        let mut stream = resp.bytes_stream();
        let mut odd: Option<u8> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("text-to-speech stream: {e}"))?;
            let mut bytes = Vec::with_capacity(chunk.len() + 1);
            if let Some(b) = odd.take() {
                bytes.push(b);
            }
            bytes.extend_from_slice(&chunk);
            if bytes.len() % 2 == 1 {
                odd = bytes.pop();
            }
            let samples = resample.process(&audio::samples(&bytes));
            if !samples.is_empty() && out.send(samples).await.is_err() {
                // Cancelled: stop reading, which ends the request.
                return Ok(());
            }
        }
        Ok(())
    }
}
