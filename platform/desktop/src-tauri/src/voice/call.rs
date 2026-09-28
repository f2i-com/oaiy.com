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
//! played. When the caller speaks over it, Aokie cancels the item, and the
//! rest of that reply is dropped. The greeting is not cut off: people say
//! "hello?" as a call connects; what they say is still heard and answered.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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

/// The output item being spoken: one per reply, each sentence added to it as it comes.
struct OpenItem {
    id: String,
    epoch: u64,
    said: Vec<String>,
    /// When its first and last audio went to Aokie, and how much has gone.
    first_pcm: Option<Instant>,
    last_pcm: Option<Instant>,
    samples: u64,
}

impl OpenItem {
    fn new(id: String, epoch: u64) -> Self {
        Self { id, epoch, said: Vec::new(), first_pcm: None, last_pcm: None, samples: 0 }
    }

    fn sent(&mut self, samples: usize, now: Instant) {
        self.first_pcm.get_or_insert(now);
        self.last_pcm = Some(now);
        self.samples += samples as u64;
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
    let (instructions, mut greeting) = (str_of("instructions"), str_of("greeting"));
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
    let (speak_tx, mut speak_rx) = mpsc::unbounded_channel::<Speak>();
    let speaker = {
        let (out_tx, ids, hub, engines, epoch, speaking, current_item, voice) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone(), epoch.clone(), speaking.clone(), current_item.clone(), voice.clone());
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
                while let Some(samples) = pcm_rx.recv().await {
                    if job.epoch != epoch.load(Ordering::SeqCst) {
                        cut = true;
                        break;
                    }
                    // At most a second per frame (Aokie takes up to two).
                    for chunk in samples.chunks(WIRE_RATE as usize) {
                        let _ = out_tx.send(Message::Binary(audio::bytes(chunk))).await;
                        item.sent(chunk.len(), Instant::now());
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
                    hub.emit(json!({"type": "call.said", "callId": ids.call, "itemId": item.id, "text": job.text}));
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

    // The listener: finished utterances, one at a time, to text.
    let (utter_tx, mut utter_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let transcriber = {
        let (out_tx, ids, hub, engines) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone());
        tokio::spawn(async move {
            while let Some(utterance) = utter_rx.recv().await {
                match engines.transcribe(&utterance).await {
                    Ok(text) if !text.is_empty() => {
                        let item = next_item("in");
                        let _ = out_tx.send(ids.event("formlogic.realtime.input_transcript", json!({"itemId": item, "transcript": text, "final": true}))).await;
                        hub.caller_said(&ids.call, &text);
                    }
                    Ok(_) => {}
                    Err(e) => hub.emit(json!({"type": "call.error", "callId": ids.call, "message": e})),
                }
            }
        })
    };

    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<CallCommand>();
    hub.register(&ids.call, cmd_tx);
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
                                if let Some(g) = v.get("greeting").and_then(Value::as_str).filter(|g| !g.trim().is_empty()) {
                                    greeting = g.to_string();
                                }
                                let (from, name) = hub.caller_of(&ids.call).unwrap_or_default();
                                hub.emit(json!({"type": "call.started", "callId": ids.call, "from": from, "name": name, "instructions": instructions, "greeting": greeting}));
                                if !greeting.trim().is_empty() {
                                    speak(greeting.clone(), true, None);
                                }
                            }
                            "formlogic.realtime.cancel_output" => {
                                let item = v.get("itemId").and_then(Value::as_str).unwrap_or("");
                                if current_item.lock().unwrap().as_deref() == Some(item) {
                                    cut();
                                    hub.emit(json!({"type": "call.interrupted", "callId": ids.call, "itemId": item, "playedMs": v.get("playedMs")}));
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
                                    }
                                    let _ = reply.send(Ok(json!({"ok": ok, "output": output})));
                                }
                            }
                            "formlogic.realtime.stop" => break v.get("reason").and_then(Value::as_str).unwrap_or("stopped").to_string(),
                            _ => {}
                        }
                    }
                    Message::Binary(bytes) => {
                        if !begun {
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
                                Heard::Started => {
                                    // Over the greeting the caller is heard, not obeyed: it plays on.
                                    if !quiet {
                                        let _ = out_tx.send(ids.event("formlogic.realtime.speech_started", json!({}))).await;
                                    }
                                    hub.emit(json!({"type": "call.speech_started", "callId": ids.call}));
                                }
                                Heard::Utterance(audio) => {
                                    let _ = utter_tx.send(audio);
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
}
