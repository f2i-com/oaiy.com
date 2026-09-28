//! One live call: Aokie's realtime stream (`formlogic.realtime.*`, caller
//! audio in as 24 kHz PCM16, our speech out the same way), what the agent in
//! the app says, and the call's tools.
//!
//! The caller's speech is found by [`Detector`], turned into text, sent to
//! Aokie (it keeps the transcript and checks appointment agreements against
//! it) and to the app, whose agent answers: each sentence it writes is spoken
//! as an output item, one after another. When the caller speaks over it, Aokie
//! cancels the item, and the rest of that reply is dropped.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

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
    item: String,
    text: String,
    epoch: u64,
    /// After it is spoken: hang up (the finish_call tool call it answers).
    then_hangup: Option<String>,
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
    let voice: Option<String> = None;

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

    // The speaker: queued texts spoken one at a time. A new epoch drops the rest.
    let epoch = Arc::new(AtomicU64::new(0));
    let speaking_out = Arc::new(AtomicBool::new(false));
    let current_item: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));
    let (speak_tx, mut speak_rx) = mpsc::unbounded_channel::<SpeakJob>();
    let speaker = {
        let (out_tx, ids, hub, engines, epoch, speaking_out, current_item, voice) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone(), epoch.clone(), speaking_out.clone(), current_item.clone(), voice.clone());
        tokio::spawn(async move {
            while let Some(job) = speak_rx.recv().await {
                if job.epoch != epoch.load(Ordering::SeqCst) {
                    continue;
                }
                *current_item.lock().unwrap() = Some(job.item.clone());
                speaking_out.store(true, Ordering::SeqCst);
                let _ = out_tx.send(ids.event("formlogic.realtime.output_item_started", json!({"itemId": job.item}))).await;
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
                    }
                }
                drop(pcm_rx);
                let spoken = synth.await.unwrap_or_else(|e| Err(e.to_string()));
                cut |= job.epoch != epoch.load(Ordering::SeqCst);
                if let Err(e) = &spoken {
                    hub.emit(json!({"type": "call.error", "callId": ids.call, "message": e}));
                }
                if !cut && spoken.is_ok() {
                    let _ = out_tx.send(ids.event("formlogic.realtime.output_transcript", json!({"itemId": job.item, "transcript": job.text, "final": true}))).await;
                    hub.emit(json!({"type": "call.said", "callId": ids.call, "itemId": job.item, "text": job.text}));
                }
                // Always: Aokie holds a cancelled item open until its exact item_done.
                let _ = out_tx.send(ids.event("formlogic.realtime.output_item_done", json!({"itemId": job.item}))).await;
                speaking_out.store(false, Ordering::SeqCst);
                *current_item.lock().unwrap() = None;
                if let (Some(tool_call_id), false) = (job.then_hangup, cut) {
                    let _ = out_tx.send(ids.event("formlogic.realtime.hangup_requested", json!({"toolCallId": tool_call_id, "responseId": format!("resp_{}", job.item), "itemId": job.item}))).await;
                }
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
    let speak = |text: String, then_hangup: Option<String>| -> String {
        let item = next_item("out");
        let _ = speak_tx.send(SpeakJob { item: item.clone(), text, epoch: epoch.load(Ordering::SeqCst), then_hangup });
        item
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
                                    speak(greeting.clone(), None);
                                }
                            }
                            "formlogic.realtime.cancel_output" => {
                                let item = v.get("itemId").and_then(Value::as_str).unwrap_or("");
                                if current_item.lock().unwrap().as_deref() == Some(item) {
                                    epoch.fetch_add(1, Ordering::SeqCst);
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
                                        speak(goodbye, Some(id));
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
                        detector.speaking_out = speaking_out.load(Ordering::SeqCst);
                        for heard in detector.push(&audio::samples(&bytes)) {
                            match heard {
                                Heard::Started => {
                                    let _ = out_tx.send(ids.event("formlogic.realtime.speech_started", json!({}))).await;
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
                        let _ = reply.send(if text.is_empty() { Err("nothing to say".into()) } else if !begun { Err("the call has not begun".into()) } else { Ok(speak(text, None)) });
                    }
                    CallCommand::Hush => {
                        epoch.fetch_add(1, Ordering::SeqCst);
                    }
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
