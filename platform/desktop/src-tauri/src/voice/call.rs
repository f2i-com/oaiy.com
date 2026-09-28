//! One live call: Aokie's realtime stream (`formlogic.realtime.*`, caller
//! audio in as 24 kHz PCM16, our speech out the same way), what the agent in
//! the app says, and the call's tools.
//!
//! The caller's speech is found by [`Detector`], turned into text, sent to
//! Aokie (it keeps the transcript and checks appointment agreements against
//! it) and to the app, whose agent answers: each sentence it writes is spoken
//! as it comes, into one output item for the reply. Aokie plays one item at a
//! time and drops what is left of a playing item when another starts, so a
//! reply's sentences share an item, and a new item waits until the last has
//! played.
//!
//! When the caller speaks over us, we do not stop at once: a cough or an
//! "mm-hmm" should not cut a sentence off. We stop when they have spoken long
//! enough to mean it, or, when they stop sooner, once their words are heard
//! and are more than an acknowledgement. Aokie does the stopping: told the
//! caller speaks (`speech_started`), it cancels the item, and the rest of that
//! reply is dropped. The greeting is not cut off: people say "hello?" as a
//! call connects; what they say is still heard and answered.
//!
//! The app is told when things were said, in milliseconds since the call began.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::audio::{self, Detector, Heard};
use super::engines::{Engines, WIRE_RATE};
use super::{VoiceHub, DESTINATION};

/// What the app asks of a live call.
pub enum CallCommand {
    /// Speak `text`, after what is queued. Answers with its item's id.
    Say { text: String, reply: oneshot::Sender<Result<String, String>> },
    /// One of Aokie's call tools (`request_appointment`, `lookup_business_data`); answers with what it returned.
    Tool { name: String, arguments: Value, reply: oneshot::Sender<Result<Value, String>> },
    /// Say goodbye, then hang up.
    Finish { goodbye: String, reply: oneshot::Sender<Result<Value, String>> },
    /// Stop speaking: what plays is cut, what is queued is dropped.
    Hush,
}

/// A text for the speaker, from a reply given before any `epoch` change.
struct SpeakJob {
    text: String,
    epoch: u64,
    /// The greeting: the caller does not cut it off.
    greeting: bool,
    /// After it is spoken: hang up (the finish_call tool call it answers).
    then_hangup: Option<String>,
}

/// What the speaker is told: a text, or to look again (a reply was cut).
enum Speak {
    Job(SpeakJob),
    Wake,
}

/// How long after an item's audio should have played Aokie is sure to be done with it.
const PLAYOUT_MARGIN: Duration = Duration::from_millis(250);

/// How long the transcriber waits for a cut it asked Aokie for.
const CUT_WAIT: Duration = Duration::from_millis(500);

/// Milliseconds since the call began (`formlogic.realtime.begin`): when things were said, for the app.
#[derive(Clone, Default)]
struct Clock(Arc<OnceLock<Instant>>);

impl Clock {
    fn start(&self) {
        let _ = self.0.set(Instant::now());
    }

    fn ms(&self, t: Instant) -> u64 {
        self.0.get().map_or(0, |begun| t.saturating_duration_since(*begun).as_millis() as u64)
    }
}

/// The output item being spoken: one per reply, each sentence added to it as it comes.
struct OpenItem {
    id: String,
    epoch: u64,
    said: Vec<String>,
    /// When its first and last audio went to Aokie, and how much has gone.
    first_pcm: Option<Instant>,
    last_pcm: Option<Instant>,
    samples: u64,
    /// When the audio sent so far will have been heard: Aokie plays it in order, as it comes.
    heard_until: Option<Instant>,
}

impl OpenItem {
    fn new(id: String, epoch: u64) -> Self {
        Self { id, epoch, said: Vec::new(), first_pcm: None, last_pcm: None, samples: 0, heard_until: None }
    }

    /// Audio sent: when the caller starts to hear it.
    fn sent(&mut self, samples: usize, now: Instant) -> Instant {
        self.first_pcm.get_or_insert(now);
        self.last_pcm = Some(now);
        self.samples += samples as u64;
        let from = self.heard_until.map_or(now, |t| t.max(now));
        self.heard_until = Some(from + Duration::from_micros(samples as u64 * 1_000_000 / WIRE_RATE as u64));
        from
    }

    /// When Aokie will have played all of it: its length after the first audio,
    /// or later when audio came slower than it plays.
    fn played_by(&self) -> Instant {
        let now = Instant::now();
        let length = Duration::from_millis(self.samples * 1000 / WIRE_RATE as u64);
        let by_length = self.first_pcm.map_or(now, |t| t + length);
        by_length.max(self.last_pcm.unwrap_or(now)) + PLAYOUT_MARGIN
    }
}

/// Whether we are heard now (the detector asks more of the caller then), and until when the caller cannot cut in.
#[derive(Default)]
struct Speaking {
    /// While an item is open: until it is closed; after, until its audio has played.
    open: bool,
    audible_until: Option<Instant>,
    /// The greeting's end: no barge-in before it.
    quiet_until: Option<Instant>,
}

impl Speaking {
    fn out(&self, now: Instant) -> bool {
        self.open || self.audible_until.is_some_and(|t| now < t)
    }

    fn quiet(&self, now: Instant) -> bool {
        self.quiet_until.is_some_and(|t| now < t)
    }
}

/// The utterance being heard, and how it stands with our voice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Hearing {
    /// Our voice was audible when they began.
    over: bool,
    /// It may yet stop us: begun over us, not over the greeting, and it has not yet.
    may_cut: bool,
    /// It has stopped us.
    cut: bool,
}

impl Hearing {
    /// They began, over our voice (`out`) or not, maybe over the greeting
    /// (`quiet`); and whether Aokie is told now. Over nothing it is, as ever;
    /// over us, not yet; over the greeting, never. Over the goodbye (`ending`)
    /// it is, as before: Aokie holds the hangup until their words are in.
    fn began(out: bool, quiet: bool, ending: bool) -> (Self, bool) {
        if quiet {
            return (Self { over: out, ..Self::default() }, false);
        }
        if !out || ending {
            return (Self { over: out, may_cut: false, cut: out }, true);
        }
        (Self { over: true, may_cut: true, cut: false }, false)
    }

    /// They have spoken long enough to mean it: whether Aokie is told now, and we stop.
    fn went_on(&mut self, out: bool, quiet: bool) -> bool {
        let tell = self.may_cut && out && !quiet;
        if tell {
            self.may_cut = false;
            self.cut = true;
        }
        tell
    }
}

/// A finished utterance, for the transcriber: its audio, when it was said
/// (ms since the call began), and how it stood with our voice.
struct Utterance {
    audio: Vec<i16>,
    start_ms: u64,
    end_ms: u64,
    /// Our voice was audible when they began.
    over: bool,
    /// It has stopped us already: they spoke long enough.
    cut: bool,
    /// Said over us, and short: its words decide whether we stop.
    decides: bool,
}

/// Whether the caller only acknowledged us ("mm-hmm", "yeah, okay"): one to
/// three of these words and nothing else. "Stop", "wait" and "no" are not.
pub fn is_backchannel(text: &str) -> bool {
    const WORDS: [&str; 20] = ["mm", "mmm", "mhm", "hmm", "uhuh", "yeah", "yep", "yes", "ok", "okay", "right", "sure", "alright", "cool", "great", "nice", "oh", "ah", "uh", "um"];
    const PAIRS: [&str; 3] = ["uh huh", "i see", "got it"];
    // "Mm-hmm." is "mm hmm": hyphens part words, other punctuation goes.
    let words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || matches!(c, '-' | '\u{2010}' | '\u{2011}' | '\u{2013}'))
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect();
    let (mut at, mut said) = (0, 0);
    while at < words.len() {
        if words.get(at + 1).is_some_and(|next| PAIRS.contains(&format!("{} {next}", words[at]).as_str())) {
            at += 2;
        } else if WORDS.contains(&words[at].as_str()) {
            at += 1;
        } else {
            return false;
        }
        said += 1;
    }
    (1..=3).contains(&said)
}

#[derive(Clone, Debug)]
struct Ids {
    call: String,
    generation: u64,
}

impl Ids {
    fn event(&self, kind: &str, fields: Value) -> Message {
        let mut v = if fields.is_object() { fields } else { json!({}) };
        v["type"] = json!(kind);
        v["callId"] = json!(self.call);
        v["generation"] = json!(self.generation);
        Message::Text(v.to_string())
    }
}

static ITEMS: AtomicU64 = AtomicU64::new(1);
fn next_item(prefix: &str) -> String {
    format!("{prefix}_{}", ITEMS.fetch_add(1, Ordering::Relaxed))
}

/// The first message: `formlogic.realtime.start`, or why the call cannot go on.
fn read_start(text: &str) -> Result<(Ids, Value), String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("invalid start event: {e}"))?;
    if v.get("type").and_then(Value::as_str) != Some("formlogic.realtime.start") {
        return Err("the first event must be formlogic.realtime.start".into());
    }
    let call = v.get("callId").and_then(Value::as_str).filter(|s| !s.is_empty() && s.len() <= 256).ok_or("start has no callId")?;
    let generation = v.get("generation").and_then(Value::as_u64).ok_or("start has no generation")?;
    let destination = v.get("destinationOrigin").and_then(Value::as_str).unwrap_or("");
    if destination != DESTINATION {
        return Err(format!("this desktop answers calls itself, at {DESTINATION}; the call asked for {destination}"));
    }
    if v.get("sampleRate").and_then(Value::as_u64).unwrap_or(WIRE_RATE as u64) != WIRE_RATE as u64 {
        return Err(format!("only {WIRE_RATE} Hz PCM16 is supported"));
    }
    Ok((Ids { call: call.to_string(), generation }, v))
}

/// A start that only asks for a line to be said (`mode: "speak"`): Aokie's screen
/// message, a hold announcement or an apology, in OAIY's voice. The greeting is
/// said once; no agent answers, the caller is not listened to, and it is not a
/// call the app sees. Aokie knows it has been said from the item's end.
fn speaks_only(start: &Value) -> bool {
    start.get("mode").and_then(Value::as_str) == Some("speak")
}

/// Run one call to its end.
pub async fn run(socket: WebSocket, hub: VoiceHub, engines: Engines) {
    let (mut sink, mut stream) = socket.split();
    // The start, before anything else.
    let (ids, start) = loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => match read_start(&text) {
                Ok(s) => break s,
                Err(e) => {
                    let _ = sink.send(Message::Text(json!({"type": "formlogic.realtime.error", "callId": "", "generation": 0, "code": "bad_start", "message": e, "fatal": true}).to_string())).await;
                    let _ = sink.close().await;
                    hub.emit(json!({"type": "call.error", "message": e}));
                    return;
                }
            },
            Some(Ok(Message::Ping(p))) => {
                let _ = sink.send(Message::Pong(p)).await;
            }
            _ => return,
        }
    };
    let str_of = |k: &str| start.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let (mut instructions, mut greeting) = (str_of("instructions"), str_of("greeting"));
    // What Aokie knows of the call (a newer Aokie sends it): which way it goes, who with,
    // and on a call it placed, why and its first words. Absent: the plugin's events say who.
    let (direction, start_from, start_name) = (str_of("direction"), str_of("from"), str_of("callerName"));
    let (purpose, opening_line) = (str_of("purpose"), str_of("openingLine"));
    let speak_only = speaks_only(&start);
    // The voice chosen for calls (a clip in the voices folder; see `voices`).
    let voice: Option<String> = super::voices::chosen();

    // Everything we send goes through one writer.
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(256);
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if sink.send(m).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });
    let _ = out_tx.send(ids.event("formlogic.realtime.ready", json!({"destinationOrigin": DESTINATION}))).await;

    // The speaker: queued texts spoken one at a time, a reply's into one item. A new epoch drops the rest.
    let epoch = Arc::new(AtomicU64::new(0));
    let speaking = Arc::new(Mutex::new(Speaking::default()));
    let current_item: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let clock = Clock::default();
    let (speak_tx, mut speak_rx) = mpsc::unbounded_channel::<Speak>();
    let speaker = {
        let (out_tx, ids, hub, engines, epoch, speaking, current_item, voice, clock) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone(), epoch.clone(), speaking.clone(), current_item.clone(), voice.clone(), clock.clone());
        tokio::spawn(async move {
            // Close the open item: what it said, then done (Aokie holds even a cancelled item open until its exact item_done).
            let close = |item: OpenItem, cut: bool| {
                let (out_tx, ids, speaking, current_item) = (out_tx.clone(), ids.clone(), speaking.clone(), current_item.clone());
                async move {
                    if !cut && !item.said.is_empty() {
                        let _ = out_tx.send(ids.event("formlogic.realtime.output_transcript", json!({"itemId": item.id, "transcript": item.said.join(" "), "final": true}))).await;
                    }
                    let _ = out_tx.send(ids.event("formlogic.realtime.output_item_done", json!({"itemId": item.id}))).await;
                    *current_item.lock().unwrap() = None;
                    let mut s = speaking.lock().unwrap();
                    s.open = false;
                    // A cut item was flushed: it is heard no more, and the next may start now.
                    s.audible_until = (!cut).then(|| item.played_by());
                    if cut {
                        s.quiet_until = None;
                    }
                    s.audible_until
                }
            };
            let mut open: Option<OpenItem> = None;
            // When the last closed item has played: the next waits for it.
            let mut played_by: Option<Instant> = None;
            loop {
                // A cut reply: its item is done now.
                if let Some(item) = open.take_if(|i| i.epoch != epoch.load(Ordering::SeqCst)) {
                    played_by = close(item, true).await;
                }
                // While an item is open, more of the reply goes into it for as long as it still plays.
                let message = match &open {
                    Some(item) => match tokio::time::timeout(item.played_by().saturating_duration_since(Instant::now()), speak_rx.recv()).await {
                        Ok(m) => m,
                        Err(_) => {
                            played_by = close(open.take().unwrap(), false).await;
                            continue;
                        }
                    },
                    None => speak_rx.recv().await,
                };
                let job = match message {
                    Some(Speak::Job(job)) => job,
                    Some(Speak::Wake) => continue,
                    None => break,
                };
                if job.epoch != epoch.load(Ordering::SeqCst) {
                    continue;
                }
                if open.is_none() {
                    // Aokie plays one item at a time: a new one waits until the last has played.
                    if let Some(t) = played_by.take() {
                        tokio::time::sleep_until(t.into()).await;
                    }
                    if job.epoch != epoch.load(Ordering::SeqCst) {
                        continue;
                    }
                    let id = next_item("out");
                    *current_item.lock().unwrap() = Some(id.clone());
                    speaking.lock().unwrap().open = true;
                    let _ = out_tx.send(ids.event("formlogic.realtime.output_item_started", json!({"itemId": id}))).await;
                    open = Some(OpenItem::new(id, job.epoch));
                }
                let item = open.as_mut().unwrap();
                if job.greeting {
                    speaking.lock().unwrap().quiet_until = Some(Instant::now() + Duration::from_secs(60));
                }
                let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<i16>>(64);
                let synth = {
                    let (engines, text, voice) = (engines.clone(), job.text.clone(), voice.clone());
                    tokio::spawn(async move { engines.speak(&text, voice.as_deref(), &pcm_tx).await })
                };
                let mut cut = false;
                // When the caller starts to hear this sentence: after what went before it in the item.
                let mut heard_from: Option<Instant> = None;
                while let Some(samples) = pcm_rx.recv().await {
                    if job.epoch != epoch.load(Ordering::SeqCst) {
                        cut = true;
                        break;
                    }
                    // At most a second per frame (Aokie takes up to two).
                    for chunk in samples.chunks(WIRE_RATE as usize) {
                        let _ = out_tx.send(Message::Binary(audio::bytes(chunk))).await;
                        let from = item.sent(chunk.len(), Instant::now());
                        heard_from.get_or_insert(from);
                    }
                }
                drop(pcm_rx);
                let spoken = synth.await.unwrap_or_else(|e| Err(e.to_string()));
                cut |= job.epoch != epoch.load(Ordering::SeqCst);
                if let Err(e) = &spoken {
                    hub.emit(json!({"type": "call.error", "callId": ids.call, "message": e}));
                }
                if job.greeting {
                    speaking.lock().unwrap().quiet_until = (!cut).then(|| item.played_by() - PLAYOUT_MARGIN);
                }
                if cut {
                    played_by = close(open.take().unwrap(), true).await;
                    continue;
                }
                if spoken.is_ok() {
                    item.said.push(job.text.clone());
                    let until = item.heard_until.unwrap_or_else(Instant::now);
                    let from = heard_from.unwrap_or(until);
                    hub.emit(json!({"type": "call.said", "callId": ids.call, "itemId": item.id, "text": job.text, "startMs": clock.ms(from), "endMs": clock.ms(until)}));
                }
                // The goodbye: its item ends with it, then Aokie hangs up once it has played.
                if let Some(tool_call_id) = job.then_hangup {
                    let item = open.take().unwrap();
                    let id = item.id.clone();
                    played_by = close(item, false).await;
                    let _ = out_tx.send(ids.event("formlogic.realtime.hangup_requested", json!({"toolCallId": tool_call_id, "responseId": format!("resp_{id}"), "itemId": id}))).await;
                }
            }
            if let Some(item) = open.take() {
                let cut = item.epoch != epoch.load(Ordering::SeqCst);
                close(item, cut).await;
            }
        })
    };

    // The listener: finished utterances, one at a time, to text. One said over
    // us, and too short to have stopped us, is decided here by its words.
    let (utter_tx, mut utter_rx) = mpsc::unbounded_channel::<Utterance>();
    let transcriber = {
        let (out_tx, ids, hub, engines, speaking, epoch) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone(), speaking.clone(), epoch.clone());
        tokio::spawn(async move {
            while let Some(utterance) = utter_rx.recv().await {
                let text = match engines.transcribe(&utterance.audio).await {
                    Ok(text) if !text.is_empty() => text,
                    Ok(_) => continue,
                    Err(e) => {
                        hub.emit(json!({"type": "call.error", "callId": ids.call, "message": e}));
                        continue;
                    }
                };
                let backchannel = utterance.over && !utterance.cut && is_backchannel(&text);
                let mut cut = utterance.cut;
                let still_out = || {
                    let s = speaking.lock().unwrap();
                    let now = Instant::now();
                    s.out(now) && !s.quiet(now)
                };
                if utterance.decides && !backchannel && still_out() {
                    // More than an "mm-hmm": Aokie cuts what still plays. The app hears of
                    // the cut before the words (it stops its agent at a cut), so wait for it.
                    let before = epoch.load(Ordering::SeqCst);
                    let _ = out_tx.send(ids.event("formlogic.realtime.speech_started", json!({}))).await;
                    let asked = Instant::now();
                    while epoch.load(Ordering::SeqCst) == before && asked.elapsed() < CUT_WAIT {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    cut = epoch.load(Ordering::SeqCst) != before;
                }
                let item = next_item("in");
                let _ = out_tx.send(ids.event("formlogic.realtime.input_transcript", json!({"itemId": item, "transcript": text, "final": true}))).await;
                let how = json!({"startMs": utterance.start_ms, "endMs": utterance.end_ms, "over": utterance.over, "cut": cut, "backchannel": backchannel});
                hub.caller_said(&ids.call, &text, how);
            }
        })
    };

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<CallCommand>();
    // A line only to be said is not a call the app answers: it is not listed, and takes no commands
    // (its sender is kept here, as a closed channel would end the call).
    let _unlisted = if speak_only {
        Some(cmd_tx)
    } else {
        hub.register(&ids.call, cmd_tx);
        None
    };
    // Speech is ready before the caller first speaks.
    {
        let (engines, hub, call) = (engines.clone(), hub.clone(), ids.call.clone());
        tokio::spawn(async move {
            if let Err(e) = engines.warm().await {
                hub.emit(json!({"type": "call.error", "callId": call, "message": e}));
            }
        });
    }

    let mut detector = Detector::new(WIRE_RATE);
    let mut hearing = Hearing::default();
    // The goodbye is being said: the call is ending.
    let mut ending = false;
    let mut begun = false;
    let mut pending_tools: HashMap<String, oneshot::Sender<Result<Value, String>>> = HashMap::new();
    // finish_call tool calls waiting for Aokie's answer: the goodbye to speak once accepted.
    let mut finishing: HashMap<String, (String, oneshot::Sender<Result<Value, String>>)> = HashMap::new();
    let speak = |text: String, greeting: bool, then_hangup: Option<String>| -> String {
        let job = next_item("say");
        let _ = speak_tx.send(Speak::Job(SpeakJob { text, epoch: epoch.load(Ordering::SeqCst), greeting, then_hangup }));
        job
    };
    // Cut what is being said: the rest of the reply is dropped, and its item ends.
    let cut = || {
        epoch.fetch_add(1, Ordering::SeqCst);
        let _ = speak_tx.send(Speak::Wake);
    };
    let reason: String = loop {
        tokio::select! {
            message = stream.next() => {
                let Some(Ok(message)) = message else { break "the stream closed".into() };
                match message {
                    Message::Text(text) => {
                        let Ok(v) = serde_json::from_str::<Value>(&text) else { continue };
                        match v.get("type").and_then(Value::as_str).unwrap_or("") {
                            "formlogic.realtime.begin" => {
                                begun = true;
                                clock.start();
                                if let Some(g) = v.get("greeting").and_then(Value::as_str).filter(|g| !g.trim().is_empty()) {
                                    greeting = g.to_string();
                                }
                                // Only a line to say: said whole, and nothing else happens.
                                if speak_only {
                                    if !greeting.trim().is_empty() {
                                        speak(greeting.clone(), true, None);
                                    }
                                    continue;
                                }
                                // The brief for this caller, which Aokie knows once the call connects: it wins over the start's.
                                if let Some(i) = v.get("instructions").and_then(Value::as_str).filter(|i| !i.trim().is_empty()) {
                                    instructions = i.to_string();
                                }
                                let (from, name) = hub.caller_of(&ids.call).unwrap_or_default();
                                let from = if from.trim().is_empty() { start_from.clone() } else { from };
                                let name = if name.trim().is_empty() { start_name.clone() } else { name };
                                // Greeted by name when we know it: the name kept for their number, else the phone's.
                                let known = super::callers::name_of(&from).or_else(|| super::callers::looks_like_name(&name).then(|| name.trim().to_string())).unwrap_or_default();
                                greeting = super::callers::personal_greeting(&greeting, &known);
                                let mut started = json!({"type": "call.started", "callId": ids.call, "from": from, "name": name, "knownName": known, "instructions": instructions, "greeting": greeting});
                                for (key, value) in [("direction", &direction), ("purpose", &purpose), ("openingLine", &opening_line)] {
                                    if !value.trim().is_empty() {
                                        started[key] = json!(value);
                                    }
                                }
                                hub.emit(started);
                                if !greeting.trim().is_empty() {
                                    speak(greeting.clone(), true, None);
                                }
                            }
                            "formlogic.realtime.cancel_output" => {
                                let item = v.get("itemId").and_then(Value::as_str).unwrap_or("");
                                if current_item.lock().unwrap().as_deref() == Some(item) {
                                    // Told before the cut: what the caller said reaches the app after it (the app stops its agent at a cut).
                                    hub.emit(json!({"type": "call.interrupted", "callId": ids.call, "itemId": item, "playedMs": v.get("playedMs"), "atMs": clock.ms(Instant::now())}));
                                    cut();
                                }
                            }
                            "formlogic.realtime.tool_result" => {
                                let id = v.get("toolCallId").and_then(Value::as_str).unwrap_or("").to_string();
                                let (ok, output) = (v.get("ok").and_then(Value::as_bool).unwrap_or(false), v.get("output").cloned().unwrap_or(Value::Null));
                                if let Some(reply) = pending_tools.remove(&id) {
                                    let _ = reply.send(Ok(json!({"ok": ok, "output": output})));
                                } else if let Some((goodbye, reply)) = finishing.remove(&id) {
                                    if ok {
                                        speak(goodbye, false, Some(id));
                                        ending = true;
                                    }
                                    let _ = reply.send(Ok(json!({"ok": ok, "output": output})));
                                }
                            }
                            "formlogic.realtime.stop" => break v.get("reason").and_then(Value::as_str).unwrap_or("stopped").to_string(),
                            _ => {}
                        }
                    }
                    Message::Binary(bytes) => {
                        // Before it begins, or when there is only a line to say, the caller is not listened to.
                        if !begun || speak_only {
                            continue;
                        }
                        let (out, quiet) = {
                            let s = speaking.lock().unwrap();
                            let now = Instant::now();
                            (s.out(now), s.quiet(now))
                        };
                        detector.speaking_out = out;
                        for heard in detector.push(&audio::samples(&bytes)) {
                            match heard {
                                Heard::Started { at_ms } => {
                                    // Over the greeting the caller is heard, not obeyed: it plays on.
                                    // Over the rest of what we say, not yet: it may be an "mm-hmm".
                                    let (next, tell) = Hearing::began(out, quiet, ending);
                                    hearing = next;
                                    if tell {
                                        let _ = out_tx.send(ids.event("formlogic.realtime.speech_started", json!({}))).await;
                                    }
                                    hub.emit(json!({"type": "call.speech_started", "callId": ids.call, "atMs": at_ms, "over": out}));
                                }
                                Heard::Sustained => {
                                    // They mean to cut in: Aokie cancels what plays, and tells us (`cancel_output`).
                                    if hearing.went_on(out, quiet) {
                                        let _ = out_tx.send(ids.event("formlogic.realtime.speech_started", json!({}))).await;
                                    }
                                }
                                Heard::Utterance { audio, start_ms, end_ms } => {
                                    let _ = utter_tx.send(Utterance { audio, start_ms, end_ms, over: hearing.over, cut: hearing.cut, decides: hearing.may_cut });
                                }
                                Heard::Nothing => {}
                            }
                        }
                    }
                    Message::Ping(p) => {
                        let _ = out_tx.send(Message::Pong(p)).await;
                    }
                    Message::Close(_) => break "the call ended".into(),
                    Message::Pong(_) => {}
                }
            }
            command = cmd_rx.recv() => {
                let Some(command) = command else { break "the desktop stopped".into() };
                match command {
                    CallCommand::Say { text, reply } => {
                        let text = text.trim().to_string();
                        // Speaking again after a goodbye the caller spoke over: the call goes on,
                        // and an "mm-hmm" is talked over again rather than taken as a word to stop for.
                        if !text.is_empty() {
                            ending = false;
                        }
                        let _ = reply.send(if text.is_empty() { Err("nothing to say".into()) } else if !begun { Err("the call has not begun".into()) } else { Ok(speak(text, false, None)) });
                    }
                    CallCommand::Hush => cut(),
                    CallCommand::Tool { name, arguments, reply } => {
                        if !matches!(name.as_str(), "request_appointment" | "lookup_business_data") {
                            let _ = reply.send(Err(format!("no call tool called {name}")));
                            continue;
                        }
                        let id = next_item("tool");
                        pending_tools.insert(id.clone(), reply);
                        let _ = out_tx.send(ids.event("formlogic.realtime.tool_call", json!({"toolCallId": id, "name": name, "arguments": arguments}))).await;
                    }
                    CallCommand::Finish { goodbye, reply } => {
                        let id = next_item("tool");
                        finishing.insert(id.clone(), (if goodbye.trim().is_empty() { "Thanks for calling. Goodbye!".into() } else { goodbye }, reply));
                        let _ = out_tx.send(ids.event("formlogic.realtime.tool_call", json!({"toolCallId": id, "name": "finish_call", "arguments": {}}))).await;
                    }
                }
            }
        }
    };

    hub.unregister(&ids.call);
    hub.emit(json!({"type": "call.ended", "callId": ids.call, "reason": reason}));
    for (_, reply) in pending_tools.drain() {
        let _ = reply.send(Err("the call ended".into()));
    }
    for (_, (_, reply)) in finishing.drain() {
        let _ = reply.send(Err("the call ended".into()));
    }
    drop(speak_tx);
    drop(utter_tx);
    let _ = transcriber.await;
    epoch.fetch_add(1, Ordering::SeqCst);
    let _ = speaker.await;
    drop(out_tx);
    let _ = writer.await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_to_say_is_asked_for_by_the_start() {
        let start = |mode: &str| serde_json::from_str::<Value>(&format!(r#"{{"type":"formlogic.realtime.start","mode":"{mode}"}}"#)).unwrap();
        assert!(speaks_only(&start("speak")));
        assert!(!speaks_only(&start("call")));
        assert!(!speaks_only(&json!({"type": "formlogic.realtime.start"})));
    }

    #[test]
    fn a_call_starts_only_for_this_desktop() {
        let ok = format!(r#"{{"type":"formlogic.realtime.start","callId":"call_1","generation":3,"destinationOrigin":"{DESTINATION}","sampleRate":24000,"greeting":"Hi"}}"#);
        let (ids, v) = read_start(&ok).unwrap();
        assert_eq!((ids.call.as_str(), ids.generation), ("call_1", 3));
        assert_eq!(v["greeting"], "Hi");
        assert!(read_start(r#"{"type":"formlogic.realtime.start","callId":"c","generation":1,"destinationOrigin":"https://api.openai.com"}"#).unwrap_err().contains("answers calls itself"));
        assert!(read_start(r#"{"type":"formlogic.realtime.begin"}"#).is_err());
        let event = ids.event("formlogic.realtime.ready", json!({"destinationOrigin": DESTINATION}));
        let Message::Text(t) = event else { panic!() };
        let v: Value = serde_json::from_str(&t).unwrap();
        assert_eq!((v["type"].as_str(), v["callId"].as_str(), v["generation"].as_u64()), (Some("formlogic.realtime.ready"), Some("call_1"), Some(3)));
    }

    #[test]
    fn an_acknowledgement_is_a_backchannel() {
        for said in ["Mm-hmm.", "mm", "Mmm", "mhm", "Hmm?", "Uh-huh.", "uh huh", "Uhuh", "Yeah.", "yep", "Yes", "OK", "O.K.", "Okay, okay.", "Right.", "Sure", "Alright", "Cool", "Great!", "Nice", "Oh", "Ah", "Uh", "Um", "I see.", "Got it.", "Yeah, got it"] {
            assert!(is_backchannel(said), "{said:?} is an acknowledgement");
        }
        // Up to three.
        assert!(is_backchannel("Yeah, yeah, yeah."));
        assert!(is_backchannel("Uh-huh, I see, got it."));
        assert!(!is_backchannel("Yeah, yeah, yeah, yeah."));
    }

    #[test]
    fn words_that_ask_us_to_stop_are_not_a_backchannel() {
        for said in ["Stop.", "Wait", "No", "Sorry?", "Hold on.", "Hang on", "Yeah, but wait", "Okay stop", "Yes please", "What?", "", "...", "Got", "I", "Tuesday"] {
            assert!(!is_backchannel(said), "{said:?} is not an acknowledgement");
        }
    }

    #[test]
    fn speech_over_us_stops_us_only_once_it_goes_on() {
        // Over nothing: Aokie is told at once, as ever, and it is not a cut.
        let (mut h, tell) = Hearing::began(false, false, false);
        assert!(tell);
        assert!(!h.over && !h.may_cut && !h.cut);
        assert!(!h.went_on(true, false), "our voice starting after them does not make it a cut");
        assert!(!h.cut);
        // Over us: not yet; once it goes on, it stops us, once.
        let (mut h, tell) = Hearing::began(true, false, false);
        assert!(!tell);
        assert!(h.over && h.may_cut);
        assert!(h.went_on(true, false));
        assert!(h.cut && !h.may_cut);
        assert!(!h.went_on(true, false), "Aokie is told once an utterance");
        // Over us, but we went quiet before it went on: its words decide.
        let (mut h, _) = Hearing::began(true, false, false);
        assert!(!h.went_on(false, false));
        assert!(h.may_cut && !h.cut);
        // Over the greeting: heard, never obeyed.
        let (mut h, tell) = Hearing::began(true, true, false);
        assert!(!tell);
        assert!(h.over && !h.may_cut);
        assert!(!h.went_on(true, false));
        assert!(!h.cut);
        // Over the goodbye: Aokie is told at once, as before (it holds the hangup for their words).
        let (mut h, tell) = Hearing::began(true, false, true);
        assert!(tell);
        assert!(h.over && h.cut && !h.may_cut);
        assert!(!h.went_on(true, false));
    }

    #[test]
    fn a_sentence_is_heard_after_what_went_before_it() {
        let t0 = Instant::now();
        let second = WIRE_RATE as usize;
        let mut item = OpenItem::new("out_1".into(), 0);
        // A second of audio sent at once is heard from then, for a second.
        assert_eq!(item.sent(second, t0), t0);
        // More sent straight after is heard after it.
        assert_eq!(item.sent(second / 2, t0 + Duration::from_millis(10)), t0 + Duration::from_secs(1));
        assert_eq!(item.heard_until, Some(t0 + Duration::from_millis(1500)));
        // Audio that came after a pause is heard when it came.
        let late = t0 + Duration::from_secs(3);
        assert_eq!(item.sent(second, late), late);
        assert_eq!(item.heard_until, Some(late + Duration::from_secs(1)));
        let clock = Clock::default();
        assert_eq!(clock.ms(late), 0, "before the call begins, all is at 0");
    }
}
