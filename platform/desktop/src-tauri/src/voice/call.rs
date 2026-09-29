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
//! What they say over us is heard at their first pause (`SETTLE_MS`), and
//! their words end sooner than other words do (see [`Detector`]). When they
//! cut us off only to acknowledge us ("yeah, sure", long enough to have
//! stopped us), we take up the reply ourselves as they stop, from the first
//! sentence they did not hear to its end (see [`Resume`]): the app is told
//! (`call.resumed`), and does not answer the acknowledgement. An
//! acknowledgement when we have stopped talking answers us, and goes to the
//! app to be answered.
//!
//! The greeting is not said the moment the call begins: the phone has
//! answered, but the caller's handset hears the line a second or two later,
//! and what is said before then is lost. See [`Opening`]: a call that came in
//! is greeted after a settle (`greetingDelayMs`, 1.5 s unless set), or as soon
//! as the caller's own "Hello?" ends; a call we placed opens when the callee's
//! "Hello?" ends, or after a silence. The caller is heard from the start, but
//! what they say before the greeting is not answered on its own: the greeting
//! answers it (the app reads it with their next words).
//!
//! The app is told when things were said, in milliseconds since the call began.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use super::audio::{self, Detector, Heard};
use super::engines::{Engines, WIRE_RATE};
use super::{voices, VoiceHub, DESTINATION};

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
    /// Which it is, in the `Ledger`.
    job: u64,
    text: String,
    epoch: u64,
    /// The greeting: the caller does not cut it off.
    greeting: bool,
    /// After it is spoken: hang up (the finish_call tool call it answers).
    then_hangup: Option<String>,
}

impl SpeakJob {
    fn new(text: String, epoch: u64, greeting: bool, then_hangup: Option<String>) -> Self {
        Self { job: ITEMS.fetch_add(1, Ordering::Relaxed), text, epoch, greeting, then_hangup }
    }
}

/// A line given to the speaker, and where its audio went.
#[derive(Clone, Debug, PartialEq)]
struct Line {
    job: u64,
    text: String,
    epoch: u64,
    /// The greeting, and a goodbye, are never said again.
    greeting: bool,
    goodbye: bool,
    /// The item its audio went into.
    item: Option<String>,
    /// Where its audio ends in that item (ms of the item's audio), once all of it has gone.
    to_ms: Option<u64>,
}

/// The lines given to the speaker and not yet played out: the open item's, and
/// those waiting to go into it. What a cut takes (by the reply's epoch) is the
/// reply the caller cut off; an item that plays out takes its lines with it.
#[derive(Default)]
struct Ledger {
    lines: Vec<Line>,
    /// The open item, and when its first audio went to Aokie.
    item_began: Option<(String, Instant)>,
}

impl Ledger {
    fn add(&mut self, job: &SpeakJob) {
        self.lines.push(Line { job: job.job, text: job.text.clone(), epoch: job.epoch, greeting: job.greeting, goodbye: job.then_hangup.is_some(), item: None, to_ms: None });
    }

    fn line(&mut self, job: u64) -> Option<&mut Line> {
        self.lines.iter_mut().find(|l| l.job == job)
    }

    /// It is being spoken into `item`.
    fn began(&mut self, job: u64, item: &str) {
        if let Some(l) = self.line(job) {
            l.item = Some(item.to_string());
        }
    }

    /// All its audio has gone, ending `to_ms` into its item.
    fn went(&mut self, job: u64, to_ms: u64) {
        if let Some(l) = self.line(job) {
            l.to_ms = Some(to_ms);
        }
    }

    /// `item` played out: its lines are done with.
    fn played(&mut self, item: &str) {
        self.lines.retain(|l| l.item.as_deref() != Some(item));
        if self.item_began.as_ref().is_some_and(|(i, _)| i == item) {
            self.item_began = None;
        }
    }

    /// The lines of a reply given before a cut (its `epoch`), taken.
    fn take(&mut self, epoch: u64) -> Vec<Line> {
        let (taken, kept) = std::mem::take(&mut self.lines).into_iter().partition(|l| l.epoch == epoch);
        self.lines = kept;
        taken
    }

    /// How much of `item` has played by now, by this desktop's clock (when Aokie does not say).
    fn played_ms(&self, item: &str) -> u64 {
        self.item_began.as_ref().filter(|(i, _)| i == item).map_or(0, |(_, t)| t.elapsed().as_millis() as u64)
    }
}

/// A reply the caller cut off, to be said after all if their words were only an
/// acknowledgement: its item, and its lines from the first the caller did not
/// hear to its end (`from`: that line's place among the reply's lines).
#[derive(Debug, PartialEq)]
struct Resume {
    item: String,
    from: usize,
    lines: Vec<String>,
}

impl Resume {
    /// What is left of the reply in `lines` when its `item` was cut off `played_ms`
    /// into it: from the first line not played to its end (one cut off in the
    /// middle is said again whole; its start alone would not make sense). Nothing,
    /// when all of it played, or it was a goodbye. The greeting is never said again.
    fn after(lines: &[Line], item: &str, played_ms: u64) -> Option<Self> {
        if lines.iter().any(|l| l.goodbye) {
            return None;
        }
        let reply: Vec<&Line> = lines.iter().filter(|l| !l.greeting).collect();
        let heard = |l: &Line| l.item.as_deref() == Some(item) && l.to_ms.is_some_and(|to| to <= played_ms);
        let from = reply.iter().position(|l| !heard(l))?;
        Some(Self { item: item.to_string(), from, lines: reply[from..].iter().map(|l| l.text.clone()).collect() })
    }
}

/// An utterance begun over our voice, heard at a pause in it (`Heard::Paused`).
struct Settle {
    start_ms: u64,
    end_ms: u64,
    /// Its words (the transcriber waits for these, rather than hearing it again).
    words: Arc<tokio::sync::OnceCell<String>>,
    /// They are in, or could not be heard.
    heard: bool,
    /// The utterance, ended before its words were in, held while it may take up the reply it cut off.
    held: Option<Utterance>,
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

/// A caller still speaking when the greeting's settle is over is waited for,
/// but the greeting is said this long after the call begins at the latest.
const GREETING_LATEST: Duration = Duration::from_millis(3_000);
/// A call we placed: the opening line waits for the callee's "Hello?", or this long in silence.
const OPENING_SILENCE: Duration = Duration::from_millis(2_500);
/// ...and is said this long after the call begins at the latest, though they still talk.
const OPENING_LATEST: Duration = Duration::from_millis(6_000);

/// When a call's first words are said: not as it begins, as the caller's
/// handset hears the line a second or two after the phone answers. A call that
/// came in is greeted once the settle is over, or as soon as the caller's
/// "Hello?" ends if they speak first; a call we placed opens as the callee's
/// "Hello?" ends, or after a silence. Someone still speaking is waited for,
/// until `latest`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Opening {
    /// Said from now on, once the caller is not speaking.
    settle: Instant,
    /// Said now, though they still speak.
    latest: Instant,
    /// The caller has said something: said as soon as they are quiet.
    heard: bool,
}

impl Opening {
    /// The wait for a call begun at `begun`: one we placed, or one that came
    /// in with a settle of `delay`. None when the greeting is said at once.
    fn after(begun: Instant, outbound: bool, delay: Duration) -> Option<Self> {
        if outbound {
            return Some(Self { settle: begun + OPENING_SILENCE, latest: begun + OPENING_LATEST, heard: false });
        }
        (!delay.is_zero()).then(|| Self { settle: begun + delay, latest: begun + delay.max(GREETING_LATEST), heard: false })
    }

    /// Whether it is said now, with the caller speaking or not.
    fn due(&self, now: Instant, caller_speaking: bool) -> bool {
        now >= self.latest || (!caller_speaking && (self.heard || now >= self.settle))
    }

    /// When to look again, if nothing is heard meanwhile.
    fn next_look(&self, now: Instant) -> Instant {
        if now < self.settle {
            self.settle
        } else {
            self.latest
        }
    }
}

/// How long a call's greeting waits after it begins: as the begin or start
/// event asks (`greetingDelayMs`: a phone that knows its line may say), else
/// as set on this desktop (`configured`); kept to 0 to 5 s.
fn greeting_delay(begin: &Value, start: &Value, configured: u64) -> Duration {
    let asked = |v: &Value| v.get("greetingDelayMs").and_then(Value::as_f64);
    let ms = asked(begin).or_else(|| asked(start)).map_or(configured.min(voices::MAX_GREETING_DELAY_MS), voices::clamp_greeting_delay);
    Duration::from_millis(ms)
}

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
    /// Begun before the greeting was said (a "Hello?" as the line opened): the greeting answers it.
    early: bool,
    /// Its words, heard at a pause in it: not heard again.
    words: Option<Arc<tokio::sync::OnceCell<String>>>,
    /// Only an acknowledgement of a reply it cut off, which went on.
    resumed: bool,
}

/// The words an acknowledgement is made of, one at a time...
const ACK_WORDS: [&str; 20] = ["mm", "mmm", "mhm", "hmm", "uhuh", "yeah", "yep", "yes", "ok", "okay", "right", "sure", "alright", "cool", "great", "nice", "oh", "ah", "uh", "um"];
/// ...or two.
const ACK_PAIRS: [&str; 3] = ["uh huh", "i see", "got it"];
/// Said quickly, these acknowledge too.
const MORE_ACK_PAIRS: [&str; 2] = ["of course", "go on"];
/// "Yeah, sure. Yeah, of course, yes.": any number of acknowledgements, said within this.
const ACK_WITHIN_MS: u64 = 1_500;

/// How many acknowledgements `text` is ("mm-hmm", "yeah", "got it": one of these
/// words, or a pair of them), or None when it says anything else.
fn acknowledgements(text: &str, pairs: &[&str]) -> Option<usize> {
    // "Mm-hmm." is "mm hmm": hyphens part words, other punctuation goes.
    let words: Vec<String> = text
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || matches!(c, '-' | '\u{2010}' | '\u{2011}' | '\u{2013}'))
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect();
    let (mut at, mut said) = (0, 0);
    while at < words.len() {
        if words.get(at + 1).is_some_and(|next| pairs.contains(&format!("{} {next}", words[at]).as_str())) {
            at += 2;
        } else if ACK_WORDS.contains(&words[at].as_str()) {
            at += 1;
        } else {
            return None;
        }
        said += 1;
    }
    Some(said)
}

/// Whether the caller only acknowledged us ("mm-hmm", "yeah, okay"): one to
/// three of these words and nothing else. "Stop", "wait" and "no" are not.
pub fn is_backchannel(text: &str) -> bool {
    acknowledgements(text, &ACK_PAIRS).is_some_and(|n| (1..=3).contains(&n))
}

/// Whether words said over us, from their first voice to their last in `span_ms`,
/// only acknowledged us: a backchannel, or short affirmatives alone said quickly
/// ("Yeah, sure. Yeah, of course, yes.", "Of course, go on."). "Yes please" asks for
/// something, and "yeah, but wait" stops us.
pub fn acknowledges(text: &str, span_ms: u64) -> bool {
    let pairs = [ACK_PAIRS.as_slice(), MORE_ACK_PAIRS.as_slice()].concat();
    is_backchannel(text) || (span_ms <= ACK_WITHIN_MS && acknowledgements(text, &pairs).is_some_and(|n| n > 0))
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
    let (sink, stream) = socket.split();
    run_on(sink, stream, hub, engines).await;
}

/// One call, over the two halves of Aokie's stream: what goes to Aokie (`sink`) and what comes from it.
async fn run_on<Si, St, E>(mut sink: Si, mut stream: St, hub: VoiceHub, engines: Engines)
where
    Si: Sink<Message> + Unpin + Send + 'static,
    St: Stream<Item = Result<Message, E>> + Unpin,
{
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
    // A call we placed opens when the callee has said "Hello?" (see `Opening`).
    let outbound = direction.trim().eq_ignore_ascii_case("outbound");
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
    // What the speaker has been given and not yet played out (see `Ledger`).
    let ledger = Arc::new(Mutex::new(Ledger::default()));
    let (speak_tx, mut speak_rx) = mpsc::unbounded_channel::<Speak>();
    let speaker = {
        let (out_tx, ids, hub, engines, epoch, speaking, current_item, voice, clock, ledger) = (out_tx.clone(), ids.clone(), hub.clone(), engines.clone(), epoch.clone(), speaking.clone(), current_item.clone(), voice.clone(), clock.clone(), ledger.clone());
        tokio::spawn(async move {
            // Close the open item: what it said, then done (Aokie holds even a cancelled item open until its exact item_done).
            let close = |item: OpenItem, cut: bool| {
                let (out_tx, ids, speaking, current_item, ledger) = (out_tx.clone(), ids.clone(), speaking.clone(), current_item.clone(), ledger.clone());
                async move {
                    if !cut && !item.said.is_empty() {
                        let _ = out_tx.send(ids.event("formlogic.realtime.output_transcript", json!({"itemId": item.id, "transcript": item.said.join(" "), "final": true}))).await;
                    }
                    let _ = out_tx.send(ids.event("formlogic.realtime.output_item_done", json!({"itemId": item.id}))).await;
                    // Played out, its lines are done with (a cut item's went with the cut).
                    if !cut {
                        ledger.lock().unwrap().played(&item.id);
                    }
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
                ledger.lock().unwrap().began(job.job, &item.id);
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
                        let first = item.first_pcm.is_none();
                        let from = item.sent(chunk.len(), Instant::now());
                        heard_from.get_or_insert(from);
                        if first {
                            ledger.lock().unwrap().item_began = Some((item.id.clone(), from));
                        }
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
                    ledger.lock().unwrap().went(job.job, item.samples * 1000 / WIRE_RATE as u64);
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
                // Heard at its pause already (or being heard): those words, not heard again.
                let heard = match &utterance.words {
                    Some(words) => words.get_or_try_init(|| engines.transcribe(&utterance.audio)).await.cloned(),
                    None => engines.transcribe(&utterance.audio).await,
                };
                let text = match heard {
                    Ok(text) if !text.is_empty() => text,
                    Ok(_) => continue,
                    Err(e) => {
                        hub.emit(json!({"type": "call.error", "callId": ids.call, "message": e}));
                        continue;
                    }
                };
                // Said before the greeting: not answered on its own (the greeting answers it), but
                // read with the caller's next words, as an "mm-hmm" is. An acknowledgement over us
                // while we still speak is one too; once we have stopped, it answers us (after "Is
                // that all right?", "Yeah, sure." is an answer), and it is answered.
                let talking = speaking.lock().unwrap().out(Instant::now());
                let acknowledged = utterance.over && !utterance.cut && talking && acknowledges(&text, utterance.end_ms.saturating_sub(utterance.start_ms));
                let backchannel = utterance.early || utterance.resumed || acknowledged;
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
                let mut how = json!({"startMs": utterance.start_ms, "endMs": utterance.end_ms, "over": utterance.over, "cut": cut, "backchannel": backchannel});
                if utterance.early {
                    how["beforeGreeting"] = json!(true);
                }
                // It stopped us for a moment, and we went on (`call.resumed`): said over us, not a cut.
                if utterance.resumed {
                    how["resumed"] = json!(true);
                }
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
    // The utterance being heard began before the greeting was said.
    let mut early = false;
    // The greeting, waiting for the line to open (see `Opening`), and what is to be said after it meanwhile.
    let mut opening: Option<Opening> = None;
    let mut after_greeting: Vec<SpeakJob> = Vec::new();
    // The goodbye is being said: the call is ending.
    let mut ending = false;
    let mut begun = false;
    let mut pending_tools: HashMap<String, oneshot::Sender<Result<Value, String>>> = HashMap::new();
    // finish_call tool calls waiting for Aokie's answer: the goodbye to speak once accepted.
    let mut finishing: HashMap<String, (String, oneshot::Sender<Result<Value, String>>)> = HashMap::new();
    // A reply the caller cut off, said after all if their words were only an acknowledgement.
    let mut resume: Option<Resume> = None;
    // What they said over us, heard at a pause in it (its words come on `words_rx`, by when it was said).
    let mut settle: Option<Settle> = None;
    let (words_tx, mut words_rx) = mpsc::unbounded_channel::<(u64, u64)>();
    // To the speaker, and kept in the ledger; a job from a reply that was cut since is dropped.
    let say = |job: SpeakJob| {
        if job.epoch == epoch.load(Ordering::SeqCst) {
            ledger.lock().unwrap().add(&job);
            let _ = speak_tx.send(Speak::Job(job));
        }
    };
    let speak = |text: String, greeting: bool, then_hangup: Option<String>| -> String {
        say(SpeakJob::new(text, epoch.load(Ordering::SeqCst), greeting, then_hangup));
        next_item("say")
    };
    // Cut what is being said: the rest of the reply is dropped, and its item ends. What it was: its lines not yet played out.
    let cut = || -> Vec<Line> {
        let was = epoch.fetch_add(1, Ordering::SeqCst);
        let _ = speak_tx.send(Speak::Wake);
        ledger.lock().unwrap().take(was)
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
                                // A contact from now on, with this as the number last seen (never a hidden caller).
                                super::contacts::saw(&from);
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
                                    // Said once the line is open: now, if the call asks for no wait.
                                    let delay = greeting_delay(&v, &start, voices::greeting_delay_ms());
                                    opening = Opening::after(Instant::now(), outbound, delay);
                                    if opening.is_none() {
                                        speak(greeting.clone(), true, None);
                                    }
                                }
                            }
                            "formlogic.realtime.cancel_output" => {
                                let item = v.get("itemId").and_then(Value::as_str).unwrap_or("");
                                if current_item.lock().unwrap().as_deref() == Some(item) {
                                    // Told before the cut: what the caller said reaches the app after it (the app stops its agent at a cut).
                                    hub.emit(json!({"type": "call.interrupted", "callId": ids.call, "itemId": item, "playedMs": v.get("playedMs"), "atMs": clock.ms(Instant::now())}));
                                    let played_ms = v.get("playedMs").and_then(Value::as_u64).unwrap_or_else(|| ledger.lock().unwrap().played_ms(item));
                                    let lines = cut();
                                    // Their voice went on over it (they meant to cut in; never a goodbye): if their
                                    // words turn out to be only an acknowledgement, what they did not hear is said after all.
                                    resume = (hearing.cut && !ending).then(|| Resume::after(&lines, item, played_ms)).flatten();
                                }
                            }
                            "formlogic.realtime.tool_result" => {
                                let id = v.get("toolCallId").and_then(Value::as_str).unwrap_or("").to_string();
                                let (ok, output) = (v.get("ok").and_then(Value::as_bool).unwrap_or(false), v.get("output").cloned().unwrap_or(Value::Null));
                                if let Some(reply) = pending_tools.remove(&id) {
                                    let _ = reply.send(Ok(json!({"ok": ok, "output": output})));
                                } else if let Some((goodbye, reply)) = finishing.remove(&id) {
                                    if ok {
                                        // Not before the greeting, while it waits for the line to open.
                                        if opening.is_some() {
                                            after_greeting.push(SpeakJob::new(goodbye, epoch.load(Ordering::SeqCst), false, Some(id)));
                                        } else {
                                            speak(goodbye, false, Some(id));
                                        }
                                        ending = true;
                                        resume = None;
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
                                    // Speaking again before the words they said last were in: those go to the app as they are.
                                    if let Some(held) = settle.take().and_then(|s| s.held) {
                                        resume = None;
                                        let _ = utter_tx.send(held);
                                    }
                                    // Over the greeting the caller is heard, not obeyed: it plays on.
                                    // Over the rest of what we say, not yet: it may be an "mm-hmm".
                                    let (next, tell) = Hearing::began(out, quiet, ending);
                                    hearing = next;
                                    early = opening.is_some();
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
                                Heard::Paused { audio, start_ms, end_ms } => {
                                    // Over us, they have paused: what they said so far is heard now, while the utterance
                                    // may yet go on. It may be all of it, and only an acknowledgement.
                                    let words = Arc::new(tokio::sync::OnceCell::new());
                                    let (engines, cell, words_tx) = (engines.clone(), words.clone(), words_tx.clone());
                                    tokio::spawn(async move {
                                        let _ = cell.get_or_try_init(|| engines.transcribe(&audio)).await;
                                        let _ = words_tx.send((start_ms, end_ms));
                                    });
                                    settle = Some(Settle { start_ms, end_ms, words, heard: false, held: None });
                                }
                                Heard::Utterance { audio, start_ms, end_ms } => {
                                    // Their "Hello?" is over: the greeting waiting for the line is said now.
                                    if let Some(o) = opening.as_mut() {
                                        o.heard = true;
                                    }
                                    let paused = settle.as_mut().filter(|s| (s.start_ms, s.end_ms) == (start_ms, end_ms));
                                    let words = paused.as_ref().map(|s| s.words.clone());
                                    let utterance = Utterance { audio, start_ms, end_ms, over: hearing.over, cut: hearing.cut, decides: hearing.may_cut, early, words, resumed: false };
                                    match paused {
                                        // It cut off a reply that may yet go on: it waits for its words, heard at its pause.
                                        Some(s) if resume.is_some() => {
                                            s.held = Some(utterance);
                                            if s.heard {
                                                let _ = words_tx.send((start_ms, end_ms));
                                            }
                                        }
                                        // To the app: a reply it cut off (if any) is the app's to go on with.
                                        _ => {
                                            resume = None;
                                            settle = None;
                                            let _ = utter_tx.send(utterance);
                                        }
                                    }
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
                        let said = if text.is_empty() {
                            Err("nothing to say".into())
                        } else if !begun {
                            Err("the call has not begun".into())
                        } else if opening.is_some() {
                            // The greeting waits for the line to open: this is said after it.
                            after_greeting.push(SpeakJob::new(text, epoch.load(Ordering::SeqCst), false, None));
                            Ok(next_item("say"))
                        } else {
                            // A new reply: it wins over one cut off that was waiting to go on.
                            resume = None;
                            Ok(speak(text, false, None))
                        };
                        let _ = reply.send(said);
                    }
                    CallCommand::Hush => {
                        resume = None;
                        cut();
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
                        // The call is ending: what was cut off is not taken up.
                        resume = None;
                        let id = next_item("tool");
                        finishing.insert(id.clone(), (if goodbye.trim().is_empty() { "Thanks for calling. Goodbye!".into() } else { goodbye }, reply));
                        let _ = out_tx.send(ids.event("formlogic.realtime.tool_call", json!({"toolCallId": id, "name": "finish_call", "arguments": {}}))).await;
                    }
                }
            }
            // The words of what they said over us, heard at a pause in it.
            Some((start_ms, end_ms)) = words_rx.recv() => {
                let Some(s) = settle.as_mut().filter(|s| (s.start_ms, s.end_ms) == (start_ms, end_ms)) else { continue };
                s.heard = true;
                // They have spoken again since that pause: a later one decides, or the end.
                if s.held.is_none() && !(detector.in_speech() && detector.voice_end_ms() == end_ms) {
                    continue;
                }
                let acknowledged = hearing.cut && s.words.get().is_some_and(|w| acknowledges(w, end_ms.saturating_sub(start_ms)));
                match resume.take() {
                    Some(r) if acknowledged => {
                        // Only an acknowledgement, and it is over: heard as one (ended now, if its pause
                        // has not ended it yet), and the reply goes on from the first line they did not hear.
                        let s = settle.take().unwrap();
                        let audio = match s.held {
                            Some(held) => held.audio,
                            None => match detector.end_now() {
                                Some(Heard::Utterance { audio, .. }) => audio,
                                _ => Vec::new(),
                            },
                        };
                        let _ = utter_tx.send(Utterance { audio, start_ms, end_ms, over: hearing.over, cut: false, decides: false, early: false, words: Some(s.words), resumed: true });
                        hearing = Hearing::default();
                        hub.emit(json!({"type": "call.resumed", "callId": ids.call, "itemId": r.item, "fromSentence": r.from, "sentences": r.lines, "atMs": clock.ms(Instant::now())}));
                        for line in r.lines {
                            speak(line, false, None);
                        }
                    }
                    pending => {
                        resume = pending;
                        // Over, and more than an acknowledgement (or not heard): the app answers it.
                        if let Some(held) = settle.as_mut().and_then(|s| s.held.take()) {
                            resume = None;
                            settle = None;
                            let _ = utter_tx.send(held);
                        }
                    }
                }
            }
            // The greeting's settle, or its latest: look again (an hour on while nothing waits; it is not polled then).
            _ = tokio::time::sleep_until(opening.map_or_else(|| Instant::now() + Duration::from_secs(3600), |o| o.next_look(Instant::now())).into()), if opening.is_some() => {}
        }
        // The line is open: the greeting, then what waited for it. A call that ends first says nothing.
        if opening.is_some_and(|o| o.due(Instant::now(), detector.in_speech())) {
            opening = None;
            // What the caller says from now on is said over the greeting (the speaker marks it too, once it starts).
            speaking.lock().unwrap().quiet_until = Some(Instant::now() + Duration::from_secs(60));
            speak(greeting.clone(), true, None);
            for job in after_greeting.drain(..) {
                say(job);
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
    use std::sync::atomic::AtomicBool;

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

    #[test]
    fn the_greeting_delay_is_the_calls_else_this_desktops_and_at_most_five_seconds() {
        let ms = |begin: Value, start: Value, configured: u64| greeting_delay(&begin, &start, configured).as_millis() as u64;
        // Nothing asked: as set here.
        assert_eq!(ms(json!({}), json!({}), 1_500), 1_500);
        assert_eq!(ms(json!({}), json!({}), 800), 800);
        assert_eq!(ms(json!({}), json!({}), 60_000), 5_000);
        // The start's, over this desktop's; the begin's, over both.
        assert_eq!(ms(json!({}), json!({"greetingDelayMs": 300}), 800), 300);
        assert_eq!(ms(json!({"greetingDelayMs": 0}), json!({"greetingDelayMs": 300}), 800), 0);
        // Kept to 0 to 5 s, in whole milliseconds.
        assert_eq!(ms(json!({"greetingDelayMs": 9_000}), json!({}), 800), 5_000);
        assert_eq!(ms(json!({"greetingDelayMs": -50}), json!({}), 800), 0);
        assert_eq!(ms(json!({"greetingDelayMs": 1234.4}), json!({}), 800), 1_234);
        // Not a number: not asked.
        assert_eq!(ms(json!({"greetingDelayMs": "soon"}), json!({"greetingDelayMs": null}), 800), 800);
    }

    #[test]
    fn a_greeting_waits_for_the_settle_or_the_callers_hello() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // No wait asked for: said at once.
        assert_eq!(Opening::after(t0, false, Duration::ZERO), None);
        // A call that came in: once the settle is over, unless the caller is speaking; then until 3 s at most.
        let o = Opening::after(t0, false, Duration::from_millis(1_500)).unwrap();
        assert!(!o.due(at(1_499), false));
        assert!(o.due(at(1_500), false));
        assert!(!o.due(at(1_500), true), "not over the caller");
        assert!(!o.due(at(2_999), true));
        assert!(o.due(at(3_000), true), "though they still talk");
        assert_eq!((o.next_look(at(0)), o.next_look(at(1_600))), (at(1_500), at(3_000)));
        // Their "Hello?" is over: said now, before the settle is.
        let heard = Opening { heard: true, ..o };
        assert!(heard.due(at(400), false));
        assert!(!heard.due(at(400), true), "they speak again");
        // A longer settle is its own latest.
        let long = Opening::after(t0, false, Duration::from_millis(4_000)).unwrap();
        assert!(!long.due(at(3_500), true));
        assert!(long.due(at(4_000), true));
        // A call we placed: the callee's "Hello?", else 2.5 s of silence; 6 s at most. Any delay is theirs to break.
        let o = Opening::after(t0, true, Duration::ZERO).unwrap();
        assert!(!o.due(at(2_499), false));
        assert!(o.due(at(2_500), false));
        assert!(!o.due(at(5_999), true));
        assert!(o.due(at(6_000), true));
        assert!(Opening { heard: true, ..o }.due(at(300), false));
    }

    // ---- A call, with a stand-in for Aokie on the other end of its stream ---------------

    /// How long each line the stand-in speech server speaks lasts, unless a test says.
    const LINE_MS: u64 = 300;

    /// A stand-in for OAIY's speech server: every line it speaks is a quiet hum
    /// (`LINE_MS` long unless a test says), and every utterance is heard as the
    /// same words ("Hello?" unless a test says), after as long as a test says
    /// (a real transcriber takes a moment).
    #[derive(Clone)]
    struct SpeechServer {
        heard: Arc<Mutex<String>>,
        hearing_ms: Arc<AtomicU64>,
        line_ms: Arc<AtomicU64>,
        /// What it was asked to speak, in order.
        spoken: Arc<Mutex<Vec<String>>>,
    }

    impl SpeechServer {
        fn new() -> Self {
            Self { heard: Arc::new(Mutex::new("Hello?".into())), hearing_ms: Arc::new(AtomicU64::new(0)), line_ms: Arc::new(AtomicU64::new(LINE_MS)), spoken: Arc::default() }
        }

        /// What the caller's utterances are heard as, from now on.
        fn hears(&self, words: &str) {
            *self.heard.lock().unwrap() = words.to_string();
        }

        fn hearing_takes(&self, ms: u64) {
            self.hearing_ms.store(ms, Ordering::SeqCst);
        }

        fn lines_last(&self, ms: u64) {
            self.line_ms.store(ms, Ordering::SeqCst);
        }

        fn spoken(&self) -> Vec<String> {
            self.spoken.lock().unwrap().clone()
        }

        async fn serve(&self) -> String {
            use axum::routing::post;
            let (line_ms, spoken) = (self.line_ms.clone(), self.spoken.clone());
            let (heard, hearing_ms) = (self.heard.clone(), self.hearing_ms.clone());
            let app = axum::Router::new()
                .route(
                    "/v1/audio/speech",
                    post(move |axum::Json(body): axum::Json<Value>| {
                        spoken.lock().unwrap().push(body["input"].as_str().unwrap_or("").to_string());
                        let line = audio::bytes(&vec![120i16; WIRE_RATE as usize * line_ms.load(Ordering::SeqCst) as usize / 1000]);
                        async move { ([("x-sample-rate", "24000")], line) }
                    }),
                )
                .route(
                    "/v1/audio/transcriptions",
                    post(move || {
                        let (text, wait) = (heard.lock().unwrap().clone(), hearing_ms.load(Ordering::SeqCst));
                        async move {
                            tokio::time::sleep(Duration::from_millis(wait)).await;
                            axum::Json(json!({"text": text}))
                        }
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://{addr}")
        }
    }

    /// Aokie's side of the stream as Aokie plays it: one item at a time, from its
    /// start to its done, as fast as it plays. Told the caller speaks
    /// (`speech_started`) while an item plays, it cancels it (`cancel_output`, with
    /// how much of it played: the audio it has had, at most the time since its
    /// first audio), and plays nothing more of it; a new item started before the
    /// cancelled one is done breaks Aokie (`fence_broken`). Everything the desktop
    /// sends is passed on, with when it came.
    fn plays(mut from_desktop: mpsc::UnboundedReceiver<Message>, to_desktop: mpsc::UnboundedSender<Message>, call: String, fence_broken: Arc<AtomicBool>) -> mpsc::UnboundedReceiver<(Instant, Message)> {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            // The item playing: its id, when its first audio came, and how much audio it has had.
            let mut playing: Option<(String, Option<Instant>, u64)> = None;
            // A cancelled item, until its done.
            let mut fenced: Option<String> = None;
            while let Some(m) = from_desktop.recv().await {
                let now = Instant::now();
                match &m {
                    Message::Text(t) => {
                        let v: Value = serde_json::from_str(t).unwrap_or_default();
                        let item = v["itemId"].as_str().unwrap_or("").to_string();
                        match v["type"].as_str().unwrap_or("") {
                            "formlogic.realtime.output_item_started" => {
                                if fenced.is_some() {
                                    fence_broken.store(true, Ordering::SeqCst);
                                }
                                playing = Some((item, None, 0));
                            }
                            "formlogic.realtime.output_item_done" => {
                                if fenced.as_deref() == Some(item.as_str()) {
                                    fenced = None;
                                }
                                if playing.as_ref().is_some_and(|(p, ..)| *p == item) {
                                    playing = None;
                                }
                            }
                            "formlogic.realtime.speech_started" => {
                                if let Some((item, first, samples)) = playing.take() {
                                    let had_ms = samples * 1000 / WIRE_RATE as u64;
                                    let played_ms = first.map_or(0, |t| now.duration_since(t).as_millis() as u64).min(had_ms);
                                    let cancel = json!({"type": "formlogic.realtime.cancel_output", "callId": call, "generation": 1, "itemId": item, "playedMs": played_ms});
                                    let _ = to_desktop.send(Message::Text(cancel.to_string()));
                                    fenced = Some(item);
                                }
                            }
                            _ => {}
                        }
                    }
                    Message::Binary(b) => {
                        if let Some((_, first, samples)) = playing.as_mut() {
                            first.get_or_insert(now);
                            *samples += b.len() as u64 / 2;
                        }
                    }
                    _ => {}
                }
                if tx.send((now, m)).is_err() {
                    break;
                }
            }
        });
        rx
    }

    /// Phone audio: `ms` of a tone (`amp` 40 is the line's hum, 6000 a voice).
    fn tone(ms: usize, amp: f32) -> Vec<i16> {
        (0..WIRE_RATE as usize * ms / 1000).map(|i| (amp * (i as f32 * 0.07).sin()) as i16).collect()
    }

    /// "Hello?": half a second of voice between the line's hum, and the pause that ends it.
    fn hello() -> Vec<i16> {
        [tone(200, 40.0), tone(500, 6_000.0), tone(900, 40.0)].concat()
    }

    const GREETING: &str = "Hi! Thanks for calling.";
    const OPENING_LINE: &str = "Hi, it's the lawn crew about Tuesday.";

    /// Aokie's side of one call, over the call's stream (its two halves as channels):
    /// what it sends the desktop, what it hears back, and what the app is told.
    struct Aokie {
        to_desktop: mpsc::UnboundedSender<Message>,
        /// What the desktop sends, as Aokie plays it (see `plays`), with when it came.
        from_desktop: mpsc::UnboundedReceiver<(Instant, Message)>,
        events: tokio::sync::broadcast::Receiver<Value>,
        hub: VoiceHub,
        call: String,
        /// When the call began (the begin was sent).
        begun: Instant,
        speech: SpeechServer,
        /// The desktop started an item before the one Aokie cancelled was done.
        fence_broken: Arc<AtomicBool>,
    }

    impl Aokie {
        /// A call started (with `fields` in its start), up to the desktop's `ready`.
        async fn start(fields: Value) -> Self {
            let speech = SpeechServer::new();
            let at = speech.serve().await;
            let hub = VoiceHub::new(Engines::at(&at, &at), |_| None);
            let events = hub.inner.events.subscribe();
            let (to_desktop, from_aokie) = mpsc::unbounded_channel::<Message>();
            let (to_aokie, from_desktop) = mpsc::unbounded_channel::<Message>();
            let stream = futures_util::stream::unfold(from_aokie, |mut rx| async move { rx.recv().await.map(|m| (Ok::<_, std::convert::Infallible>(m), rx)) });
            let sink = futures_util::sink::unfold(to_aokie, |tx, m: Message| async move { tx.send(m).map(|()| tx).map_err(|_| "Aokie hung up") });
            tokio::spawn(run_on(Box::pin(sink), Box::pin(stream), hub.clone(), Engines::at(&at, &at)));
            let call = format!("call_{}", next_item("test"));
            let fence_broken = Arc::new(AtomicBool::new(false));
            let from_desktop = plays(from_desktop, to_desktop.clone(), call.clone(), fence_broken.clone());
            let mut start = json!({"type": "formlogic.realtime.start", "callId": call, "generation": 1, "destinationOrigin": DESTINATION, "sampleRate": WIRE_RATE, "greeting": GREETING, "direction": "inbound"});
            for (key, value) in fields.as_object().cloned().unwrap_or_default() {
                start[key] = value;
            }
            to_desktop.send(Message::Text(start.to_string())).unwrap();
            let mut aokie = Self { to_desktop, from_desktop, events, hub, call, begun: Instant::now(), speech, fence_broken };
            let ready = aokie.next(Duration::from_secs(5)).await;
            assert!(matches!(&ready, Some(Message::Text(t)) if t.contains("formlogic.realtime.ready")), "not ready");
            aokie
        }

        /// The caller's audio as they speak it: 20 ms at a time, each once it has been said
        /// (the first 20 ms after now). When they began: what they say `ms` into it was heard
        /// by `began + ms`.
        fn caller_live(&self, samples: Vec<i16>) -> Instant {
            let began = Instant::now();
            let frame = Duration::from_millis(20);
            let to_desktop = self.to_desktop.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval_at((began + frame).into(), frame);
                for chunk in samples.chunks(WIRE_RATE as usize / 50) {
                    tick.tick().await;
                    if to_desktop.send(Message::Binary(audio::bytes(chunk))).is_err() {
                        break;
                    }
                }
            });
            began
        }

        /// The next output item: its id, and when its first audio came (what the desktop sent before it is skipped).
        async fn next_item_audio(&mut self, wait: Duration) -> Option<(String, Instant)> {
            let until = Instant::now() + wait;
            let mut item = None;
            loop {
                let (at, m) = tokio::time::timeout(until.saturating_duration_since(Instant::now()), self.from_desktop.recv()).await.ok().flatten()?;
                match m {
                    Message::Text(t) => {
                        let v: Value = serde_json::from_str(&t).unwrap();
                        if v["type"] == "formlogic.realtime.output_item_started" {
                            item = v["itemId"].as_str().map(str::to_string);
                        }
                    }
                    Message::Binary(_) => {
                        if let Some(item) = item.take() {
                            return Some((item, at));
                        }
                    }
                    _ => {}
                }
            }
        }

        /// Everything the desktop sends within `wait`, as Aokie's texts (audio: `{"audio": bytes}`).
        async fn sent_within(&mut self, wait: Duration) -> Vec<Value> {
            let until = Instant::now() + wait;
            let mut sent = Vec::new();
            while let Ok(Some((_, m))) = tokio::time::timeout(until.saturating_duration_since(Instant::now()), self.from_desktop.recv()).await {
                match m {
                    Message::Text(t) => sent.push(serde_json::from_str(&t).unwrap()),
                    Message::Binary(b) => sent.push(json!({"audio": b.len()})),
                    _ => {}
                }
            }
            sent
        }

        /// Every event the app is told within `wait` (and those told before, not yet read).
        async fn events_within(&mut self, wait: Duration) -> Vec<Value> {
            let until = Instant::now() + wait;
            let mut told = Vec::new();
            while let Ok(got) = tokio::time::timeout(until.saturating_duration_since(Instant::now()), self.events.recv()).await {
                match got {
                    Ok(v) => told.push(v),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }
            told
        }

        /// The call connects (with `fields` in the begin).
        fn begin(&mut self, fields: Value) {
            let mut begin = json!({"type": "formlogic.realtime.begin", "callId": self.call, "generation": 1});
            for (key, value) in fields.as_object().cloned().unwrap_or_default() {
                begin[key] = value;
            }
            self.begun = Instant::now();
            self.to_desktop.send(Message::Text(begin.to_string())).unwrap();
        }

        /// The caller's audio, all at once (the desktop hears it by its length, as it plays).
        fn caller(&self, samples: &[i16]) {
            for frame in samples.chunks(WIRE_RATE as usize / 50) {
                self.to_desktop.send(Message::Binary(audio::bytes(frame))).unwrap();
            }
        }

        /// A text for Aokie (a stop, a tool's result).
        fn send(&self, event: Value) {
            self.to_desktop.send(Message::Text(event.to_string())).unwrap();
        }

        /// What the desktop sends next, within `wait` (None: nothing, or the stream has closed).
        async fn next(&mut self, wait: Duration) -> Option<Message> {
            tokio::time::timeout(wait, self.from_desktop.recv()).await.ok().flatten().map(|(_, m)| m)
        }

        /// The first of our speech the caller would hear: how long after the call began it
        /// came, and the events before it.
        async fn first_audio(&mut self, wait: Duration) -> Option<(Duration, Vec<Value>)> {
            let until = Instant::now() + wait;
            let mut before = Vec::new();
            loop {
                match self.next(until.saturating_duration_since(Instant::now())).await? {
                    Message::Binary(_) => return Some((self.begun.elapsed(), before)),
                    Message::Text(t) => before.push(serde_json::from_str(&t).unwrap()),
                    _ => {}
                }
            }
        }

        /// The next event of `kind` sent to Aokie, within `wait`.
        async fn text(&mut self, kind: &str, wait: Duration) -> Option<Value> {
            let until = Instant::now() + wait;
            loop {
                if let Message::Text(t) = self.next(until.saturating_duration_since(Instant::now())).await? {
                    let v: Value = serde_json::from_str(&t).unwrap();
                    if v["type"] == kind {
                        return Some(v);
                    }
                }
            }
        }

        /// The next event of `kind` the app is told, within `wait`.
        async fn event(&mut self, kind: &str, wait: Duration) -> Option<Value> {
            let until = Instant::now() + wait;
            loop {
                match tokio::time::timeout(until.saturating_duration_since(Instant::now()), self.events.recv()).await.ok()? {
                    Ok(v) if v["type"] == kind => return Some(v),
                    Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => return None,
                }
            }
        }

        /// Ask the call to say `text`, as the app does.
        async fn say(&self, text: &str) -> Result<String, String> {
            let (reply, answer) = oneshot::channel();
            assert!(self.hub.command(&self.call).expect("the call is listed").send(CallCommand::Say { text: text.into(), reply }).is_ok());
            answer.await.unwrap()
        }
    }

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[tokio::test]
    async fn a_call_that_came_in_is_greeted_once_the_line_has_settled_not_before() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({}));
        // The app, asking to say something while the greeting waits: it is said after it.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(aokie.say("How can I help?").await.is_ok());
        let (at, before) = aokie.first_audio(secs(6)).await.expect("the greeting");
        assert!(at >= Duration::from_millis(1_500), "greeted {at:?} after the call began: before the caller could hear it");
        assert!(at < Duration::from_millis(3_000), "greeted {at:?} after the call began");
        assert!(before.iter().any(|v| v["type"] == "formlogic.realtime.output_item_started"), "{before:?}");
        // Its time is when it was said, not when the call began.
        let said = aokie.event("call.said", secs(5)).await.expect("the greeting, told to the app");
        assert_eq!(said["text"], GREETING);
        assert!(said["startMs"].as_u64().unwrap() >= 1_500, "{said}");
        let item = aokie.text("formlogic.realtime.output_transcript", secs(5)).await.expect("what the item said");
        assert_eq!(item["transcript"], format!("{GREETING} How can I help?"));
    }

    #[tokio::test]
    async fn a_hello_before_the_greeting_brings_it_forward_and_is_not_answered_on_its_own() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({}));
        aokie.caller(&hello());
        let (at, mut sent) = aokie.first_audio(secs(6)).await.expect("the greeting");
        assert!(at < Duration::from_millis(1_000), "greeted {at:?} after the call began: their hello ended well before the settle");
        while let Some(m) = aokie.next(Duration::from_millis(1_500)).await {
            if let Message::Text(t) = m {
                sent.push(serde_json::from_str(&t).unwrap());
            }
        }
        // Their hello is heard, and Aokie has its words, as ever...
        assert!(sent.iter().any(|v| v["type"] == "formlogic.realtime.input_transcript" && v["transcript"] == "Hello?"), "{sent:?}");
        // ...but the app reads it with their next words (as an "mm-hmm"), and does not answer it alone: the greeting did.
        let heard = aokie.event("call.caller", secs(5)).await.expect("their hello, for the app");
        assert_eq!((heard["text"].as_str(), heard["beforeGreeting"].as_bool(), heard["backchannel"].as_bool()), (Some("Hello?"), Some(true), Some(true)), "{heard}");
        assert_eq!(heard["over"], false);
        // And nothing else is said: one item, the greeting.
        assert_eq!(sent.iter().filter(|v| v["type"] == "formlogic.realtime.output_item_started").count(), 1, "{sent:?}");
    }

    #[tokio::test]
    async fn a_caller_who_talks_on_is_greeted_after_three_seconds_at_most() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({}));
        // Talking from the start, and not stopping.
        aokie.caller(&[tone(100, 40.0), tone(5_000, 6_000.0)].concat());
        let (at, _) = aokie.first_audio(secs(8)).await.expect("the greeting");
        assert!(at >= Duration::from_millis(3_000), "greeted {at:?} after the call began, over the caller");
        assert!(at < Duration::from_millis(4_500), "greeted {at:?} after the call began");
        // What they said, begun before the greeting, is read with their next words too.
        aokie.caller(&tone(1_000, 40.0));
        let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
        assert_eq!((heard["beforeGreeting"].as_bool(), heard["backchannel"].as_bool()), (Some(true), Some(true)), "{heard}");
    }

    #[tokio::test]
    async fn a_call_we_placed_opens_when_the_callee_has_said_hello() {
        let mut aokie = Aokie::start(json!({"direction": "outbound", "openingLine": OPENING_LINE, "greeting": OPENING_LINE})).await;
        aokie.begin(json!({}));
        tokio::time::sleep(Duration::from_millis(300)).await;
        aokie.caller(&hello());
        let (at, _) = aokie.first_audio(secs(8)).await.expect("the opening line");
        assert!(at >= Duration::from_millis(300), "opened {at:?} after the call began, before the callee spoke");
        assert!(at < Duration::from_millis(1_500), "opened {at:?} after the call began: the callee had said hello");
        // What the app is told, in whichever order: the opening line, and their hello (not to be answered alone).
        let (mut said, mut heard) = (None, None);
        while said.is_none() || heard.is_none() {
            let v = tokio::time::timeout(secs(5), aokie.events.recv()).await.expect("the app is told").unwrap();
            match v["type"].as_str() {
                Some("call.said") => said = Some(v),
                Some("call.caller") => heard = Some(v),
                _ => {}
            }
        }
        assert_eq!(said.unwrap()["text"], OPENING_LINE);
        let heard = heard.unwrap();
        assert_eq!((heard["beforeGreeting"].as_bool(), heard["backchannel"].as_bool()), (Some(true), Some(true)), "{heard}");
    }

    #[tokio::test]
    async fn a_call_we_placed_opens_after_two_and_a_half_seconds_of_silence() {
        let mut aokie = Aokie::start(json!({"direction": "outbound", "openingLine": OPENING_LINE, "greeting": OPENING_LINE})).await;
        aokie.begin(json!({}));
        // The line's hum is not a hello.
        aokie.caller(&tone(2_000, 40.0));
        let (at, _) = aokie.first_audio(secs(8)).await.expect("the opening line");
        assert!(at >= Duration::from_millis(2_500), "opened {at:?} after the call began");
        assert!(at < Duration::from_millis(4_500), "opened {at:?} after the call began");
    }

    #[tokio::test]
    async fn a_line_only_to_say_is_said_at_once() {
        let mut aokie = Aokie::start(json!({"mode": "speak", "greeting": "Please hold."})).await;
        // A wait asked for is not for a call already connected.
        aokie.begin(json!({"greetingDelayMs": 3_000}));
        let (at, _) = aokie.first_audio(secs(6)).await.expect("the line");
        assert!(at < Duration::from_millis(1_000), "said {at:?} after it began");
    }

    #[tokio::test]
    async fn a_call_that_ends_before_the_greeting_says_nothing() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({}));
        tokio::time::sleep(Duration::from_millis(300)).await;
        aokie.send(json!({"type": "formlogic.realtime.stop", "callId": aokie.call, "generation": 1, "reason": "the caller hung up"}));
        // Everything the desktop sends, until it closes the stream.
        let mut sent = Vec::new();
        while let Some(m) = aokie.next(secs(5)).await {
            sent.push(m);
        }
        assert!(aokie.begun.elapsed() < secs(5), "the stream was not closed");
        assert!(!sent.iter().any(|m| matches!(m, Message::Binary(_))), "spoke after the call ended");
        assert!(!sent.iter().any(|m| matches!(m, Message::Text(t) if t.contains("output_item_started"))), "{} messages", sent.len());
        // The settle has passed, and still nothing.
        tokio::time::sleep_until((aokie.begun + Duration::from_millis(1_800)).into()).await;
        let ended = aokie.event("call.ended", secs(1)).await.expect("the end, told to the app");
        assert_eq!(ended["reason"], "the caller hung up");
        while let Ok(v) = aokie.events.try_recv() {
            assert_ne!(v["type"], "call.said", "{v}");
        }
    }

    #[tokio::test]
    async fn the_greeting_waits_as_long_as_the_call_asks_kept_to_five_seconds() {
        // As the begin asks.
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({"greetingDelayMs": 400}));
        let (at, _) = aokie.first_audio(secs(6)).await.expect("the greeting");
        assert!(at >= Duration::from_millis(400) && at < Duration::from_millis(1_400), "greeted {at:?} after the call began");
        // No wait at all.
        let mut aokie = Aokie::start(json!({})).await;
        aokie.begin(json!({"greetingDelayMs": 0}));
        let (at, _) = aokie.first_audio(secs(6)).await.expect("the greeting");
        assert!(at < Duration::from_millis(1_000), "greeted {at:?} after the call began");
        // As the start asks, and at most 5 s.
        let mut aokie = Aokie::start(json!({"greetingDelayMs": 60_000})).await;
        aokie.begin(json!({}));
        let (at, _) = aokie.first_audio(secs(9)).await.expect("the greeting");
        assert!(at >= Duration::from_millis(5_000) && at < Duration::from_millis(6_500), "greeted {at:?} after the call began");
    }

    // ---- Talked over: timed as the caller speaks, 20 ms at a time ----------------------

    /// What the agent says in these calls: three lines, 1.5 s each.
    const REPLY: [&str; 3] = ["We are open from nine.", "We close at five.", "And on Sundays we rest."];
    const LINE_LONG_MS: u64 = 1_500;

    /// The call has begun, its greeting has played out, and the reply (`lines`, each
    /// `LINE_LONG_MS`) has begun to play: its item, and when its first audio came.
    async fn a_reply_playing(aokie: &mut Aokie, lines: &[&str]) -> (String, Instant) {
        aokie.speech.lines_last(LINE_LONG_MS);
        aokie.begin(json!({"greetingDelayMs": 0}));
        let (greeting, _) = aokie.next_item_audio(secs(5)).await.expect("the greeting");
        loop {
            let done = aokie.text("formlogic.realtime.output_item_done", secs(10)).await.expect("the greeting's end");
            if done["itemId"] == greeting.as_str() {
                break;
            }
        }
        for line in lines {
            aokie.say(line).await.unwrap();
        }
        aokie.next_item_audio(secs(5)).await.expect("the reply")
    }

    /// `ms` of the caller's voice, then the line's hum for long enough to end it.
    fn saying(ms: usize) -> Vec<i16> {
        [tone(ms, 6_000.0), tone(3_000, 40.0)].concat()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_caller_who_cuts_in_is_answered_soon_after_they_stop() {
        let mut aokie = Aokie::start(json!({})).await;
        // A transcriber that takes a moment, as a real one does.
        aokie.speech.hearing_takes(150);
        let (item, first) = a_reply_playing(&mut aokie, &REPLY).await;
        tokio::time::sleep_until((first + Duration::from_millis(400)).into()).await;
        aokie.speech.hears("Wait, is Saturday free?");
        let began = aokie.caller_live(saying(1_200));
        let stopped = began + Duration::from_millis(1_200);
        // The app is told of the cut, and then of their words (with no reply taken up between).
        let mut told = Vec::new();
        let heard = loop {
            let v = tokio::time::timeout(secs(5), aokie.events.recv()).await.expect("their words, for the app").unwrap();
            if v["type"] == "call.caller" {
                break v;
            }
            told.push(v);
        };
        let heard_after = stopped.elapsed();
        let cut: Vec<&Value> = told.iter().filter(|v| v["type"] == "call.interrupted").collect();
        assert_eq!(cut.len(), 1, "{told:?}");
        assert_eq!(cut[0]["itemId"], item.as_str());
        assert!(!told.iter().any(|v| v["type"] == "call.resumed"), "{told:?}");
        assert_eq!((heard["text"].as_str(), heard["cut"].as_bool(), heard["backchannel"].as_bool()), (Some("Wait, is Saturday free?"), Some(true), Some(false)), "{heard}");
        // The app answers them the moment it hears them: the next thing said is its answer.
        aokie.say("Saturday too, from nine.").await.unwrap();
        let (_, answer) = aokie.next_item_audio(secs(5)).await.expect("the answer");
        let answered = answer.duration_since(stopped);
        eprintln!("TIMING a real interruption (hearing takes 150 ms): the caller stopped; their words reached the app {} ms later, and its answer was heard {} ms later", heard_after.as_millis(), answered.as_millis());
        assert_eq!(aokie.speech.spoken(), [&[GREETING][..], &REPLY, &["Saturday too, from nine."]].concat(), "nothing of the cut reply was said again");
        assert!(!aokie.fence_broken.load(Ordering::SeqCst), "an item started before the cancelled one was done");
        // Their words end at 500 ms of quiet, and were heard at 300 ms: in by the end.
        assert!(answered < Duration::from_millis(650), "answered {answered:?} after the caller stopped");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn yeah_sure_over_the_reply_takes_it_up_again_as_they_stop_from_the_line_not_heard() {
        let mut aokie = Aokie::start(json!({})).await;
        let (item, first) = a_reply_playing(&mut aokie, &REPLY).await;
        // Into its second line: the first was heard whole.
        tokio::time::sleep_until((first + Duration::from_millis(1_700)).into()).await;
        aokie.speech.hears("Yeah, sure.");
        // Long enough over the reply (600 ms of voice) to cut it off.
        let began = aokie.caller_live(saying(800));
        let stopped = began + Duration::from_millis(800);
        let (again, heard_again) = aokie.next_item_audio(secs(5)).await.expect("the reply, taken up again");
        let after = heard_again.duration_since(stopped);
        eprintln!("TIMING yeah sure over the reply: it went on {} ms after they stopped", after.as_millis());
        assert!(after < Duration::from_millis(450), "went on {after:?} after they stopped");
        assert_ne!(again, item);
        // Aokie is told what the item said, once it has played (two lines: 3 s).
        let sent = aokie.sent_within(Duration::from_millis(3_600)).await;
        let transcript = sent.iter().find(|v| v["type"] == "formlogic.realtime.output_transcript" && v["itemId"] == again.as_str());
        assert_eq!(transcript.map(|v| v["transcript"].clone()), Some(json!("We close at five. And on Sundays we rest.")), "{sent:?}");
        assert!(!aokie.fence_broken.load(Ordering::SeqCst), "an item started before the cancelled one was done");
        // What was said: the reply, then from its second line again.
        assert_eq!(aokie.speech.spoken(), [&[GREETING][..], &REPLY, &REPLY[1..]].concat());
        // The app: the cut, the reply going on (from which line), their words as an acknowledgement, and the lines said again.
        let told = aokie.events_within(Duration::from_millis(100)).await;
        let of = |kind: &str| told.iter().filter(|v| v["type"] == kind).cloned().collect::<Vec<_>>();
        assert_eq!(of("call.interrupted").len(), 1);
        assert_eq!(of("call.interrupted")[0]["itemId"], item.as_str());
        let resumed = of("call.resumed");
        assert_eq!(resumed.len(), 1, "{told:?}");
        assert_eq!((&resumed[0]["itemId"], &resumed[0]["fromSentence"], &resumed[0]["sentences"]), (&json!(item), &json!(1), &json!(REPLY[1..])));
        let heard = of("call.caller");
        assert_eq!(heard.len(), 1, "{told:?}");
        assert_eq!((heard[0]["text"].as_str(), heard[0]["over"].as_bool(), heard[0]["cut"].as_bool(), heard[0]["backchannel"].as_bool(), heard[0]["resumed"].as_bool()), (Some("Yeah, sure."), Some(true), Some(false), Some(true), Some(true)), "{}", heard[0]);
        let said_again: Vec<&Value> = told.iter().filter(|v| v["type"] == "call.said" && v["itemId"] == again.as_str()).map(|v| &v["text"]).collect();
        assert_eq!(said_again, [&json!(REPLY[1]), &json!(REPLY[2])]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_short_mm_hmm_over_the_reply_does_not_stop_it() {
        let mut aokie = Aokie::start(json!({})).await;
        let (item, first) = a_reply_playing(&mut aokie, &REPLY).await;
        tokio::time::sleep_until((first + Duration::from_millis(400)).into()).await;
        aokie.speech.hears("Mm-hmm.");
        aokie.caller_live(saying(300));
        let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
        assert_eq!((heard["over"].as_bool(), heard["cut"].as_bool(), heard["backchannel"].as_bool()), (Some(true), Some(false), Some(true)), "{heard}");
        // The reply plays on to its end: Aokie is never told to stop it, and nothing is said again.
        let sent = aokie.sent_within(Duration::from_millis(4_500)).await;
        assert!(!sent.iter().any(|v| v["type"] == "formlogic.realtime.speech_started" || v["type"] == "formlogic.realtime.output_item_started"), "{sent:?}");
        assert!(sent.iter().any(|v| v["type"] == "formlogic.realtime.output_item_done" && v["itemId"] == item.as_str()), "{sent:?}");
        assert_eq!(aokie.speech.spoken(), [&[GREETING][..], &REPLY].concat());
        let told = aokie.events_within(Duration::from_millis(100)).await;
        assert!(!told.iter().any(|v| v["type"] == "call.interrupted" || v["type"] == "call.resumed"), "{told:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_reply_from_the_app_while_the_caller_settles_wins_over_taking_the_cut_one_up() {
        let mut aokie = Aokie::start(json!({})).await;
        let (_, first) = a_reply_playing(&mut aokie, &REPLY).await;
        tokio::time::sleep_until((first + Duration::from_millis(400)).into()).await;
        aokie.speech.hears("Yeah, sure.");
        aokie.caller_live(saying(800));
        aokie.event("call.interrupted", secs(5)).await.expect("the reply was cut");
        // The app has something new to say before their words are in.
        aokie.say("Sorry, go ahead.").await.unwrap();
        let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
        // Not taken as the cut reply's acknowledgement: the app has their words to answer.
        assert_eq!((heard["cut"].as_bool(), heard["backchannel"].as_bool(), heard.get("resumed")), (Some(true), Some(false), None), "{heard}");
        let sent = aokie.sent_within(Duration::from_millis(1_500)).await;
        assert_eq!(sent.iter().filter(|v| v["type"] == "formlogic.realtime.output_item_started").count(), 1, "only the app's reply: {sent:?}");
        assert_eq!(aokie.speech.spoken(), [&[GREETING][..], &REPLY, &["Sorry, go ahead."]].concat());
        let told = aokie.events_within(Duration::from_millis(100)).await;
        assert!(!told.iter().any(|v| v["type"] == "call.resumed"), "{told:?}");
        assert!(!aokie.fence_broken.load(Ordering::SeqCst), "an item started before the cancelled one was done");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn yeah_sure_over_the_greeting_is_heard_not_obeyed_and_the_greeting_plays_on() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.speech.lines_last(3_000);
        aokie.begin(json!({"greetingDelayMs": 0}));
        let (greeting, first) = aokie.next_item_audio(secs(5)).await.expect("the greeting");
        tokio::time::sleep_until((first + Duration::from_millis(300)).into()).await;
        aokie.speech.hears("Yeah, sure.");
        aokie.caller_live(saying(800));
        // Their words are in while it still plays: an acknowledgement, and it goes on.
        let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
        assert_eq!((heard["over"].as_bool(), heard["cut"].as_bool(), heard["backchannel"].as_bool()), (Some(true), Some(false), Some(true)), "{heard}");
        let sent = aokie.sent_within(Duration::from_millis(2_500)).await;
        assert!(!sent.iter().any(|v| v["type"] == "formlogic.realtime.speech_started" || v["type"] == "formlogic.realtime.output_item_started"), "{sent:?}");
        let said = sent.iter().find(|v| v["type"] == "formlogic.realtime.output_transcript" && v["itemId"] == greeting.as_str());
        assert_eq!(said.map(|v| v["transcript"].clone()), Some(json!(GREETING)), "{sent:?}");
        assert_eq!(aokie.speech.spoken(), [GREETING]);
        let told = aokie.events_within(Duration::from_millis(100)).await;
        assert!(!told.iter().any(|v| v["type"] == "call.interrupted" || v["type"] == "call.resumed"), "{told:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn yes_said_as_a_question_ends_answers_it_and_is_answered_soon() {
        // An outbound call's opening line asks "Is that all right?": they say yes over its last 300 ms.
        for words in ["Yeah, sure. Yeah, of course, yes.", "Yeah."] {
            let mut aokie = Aokie::start(json!({})).await;
            aokie.speech.hearing_takes(150);
            aokie.speech.lines_last(LINE_LONG_MS);
            aokie.begin(json!({"greetingDelayMs": 0}));
            let (_, first) = aokie.next_item_audio(secs(5)).await.expect("the greeting");
            tokio::time::sleep_until((first + Duration::from_millis(1_200)).into()).await;
            aokie.speech.hears(words);
            let began = aokie.caller_live(saying(1_000));
            let stopped = began + Duration::from_millis(1_000);
            // It had stopped by the time their words were in: they answer it (not set aside as an
            // acknowledgement, to be read with words that never come), and the app answers them.
            let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
            let heard_after = stopped.elapsed();
            eprintln!("TIMING {words:?} said over the end of the greeting (hearing takes 150 ms): it reached the app {} ms after they stopped, as {heard}", heard_after.as_millis());
            assert_eq!((heard["over"].as_bool(), heard["cut"].as_bool(), heard["backchannel"].as_bool()), (Some(true), Some(false), Some(false)), "{heard}");
            assert!(heard_after < Duration::from_millis(650), "{heard_after:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_goodbye_spoken_over_is_never_taken_up_again() {
        let mut aokie = Aokie::start(json!({})).await;
        aokie.speech.lines_last(LINE_LONG_MS);
        aokie.begin(json!({"greetingDelayMs": 0}));
        let (greeting, _) = aokie.next_item_audio(secs(5)).await.expect("the greeting");
        while aokie.text("formlogic.realtime.output_item_done", secs(5)).await.expect("the greeting's end")["itemId"] != greeting.as_str() {}
        // The app ends the call: Aokie accepts, and the goodbye is said.
        let (reply, _answer) = oneshot::channel();
        assert!(aokie.hub.command(&aokie.call).unwrap().send(CallCommand::Finish { goodbye: "Thanks for calling, bye!".into(), reply }).is_ok());
        let tool = aokie.text("formlogic.realtime.tool_call", secs(5)).await.expect("finish_call");
        aokie.send(json!({"type": "formlogic.realtime.tool_result", "callId": aokie.call, "generation": 1, "toolCallId": tool["toolCallId"], "ok": true, "output": {}}));
        let (_, first) = aokie.next_item_audio(secs(5)).await.expect("the goodbye");
        tokio::time::sleep_until((first + Duration::from_millis(300)).into()).await;
        aokie.speech.hears("Okay, bye.");
        aokie.caller_live(saying(800));
        // Aokie is told at once (it holds the hangup for their words); the goodbye's item ended with
        // its words, so there is nothing to cut, and nothing is taken up again.
        let heard = aokie.event("call.caller", secs(5)).await.expect("their words, for the app");
        assert_eq!((heard["over"].as_bool(), heard.get("resumed")), (Some(true), None), "{heard}");
        let sent = aokie.sent_within(Duration::from_millis(1_500)).await;
        assert!(sent.iter().any(|v| v["type"] == "formlogic.realtime.speech_started"), "{sent:?}");
        assert!(!sent.iter().any(|v| v["type"] == "formlogic.realtime.output_item_started"), "{sent:?}");
        assert_eq!(aokie.speech.spoken(), [GREETING, "Thanks for calling, bye!"]);
        let told = aokie.events_within(Duration::from_millis(100)).await;
        assert!(!told.iter().any(|v| v["type"] == "call.resumed"), "{told:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_call_that_ends_while_the_caller_settles_says_nothing_more() {
        let mut aokie = Aokie::start(json!({})).await;
        let (_, first) = a_reply_playing(&mut aokie, &REPLY).await;
        tokio::time::sleep_until((first + Duration::from_millis(400)).into()).await;
        aokie.speech.hears("Yeah, sure.");
        aokie.caller_live(saying(800));
        aokie.event("call.interrupted", secs(5)).await.expect("the reply was cut");
        aokie.send(json!({"type": "formlogic.realtime.stop", "callId": aokie.call, "generation": 1, "reason": "the caller hung up"}));
        let sent = aokie.sent_within(Duration::from_millis(1_500)).await;
        assert!(!sent.iter().any(|v| v["type"] == "formlogic.realtime.output_item_started"), "{sent:?}");
        assert_eq!(aokie.speech.spoken(), [&[GREETING][..], &REPLY].concat());
        let told = aokie.events_within(Duration::from_millis(100)).await;
        assert!(told.iter().any(|v| v["type"] == "call.ended"), "{told:?}");
        assert!(!told.iter().any(|v| v["type"] == "call.resumed"), "{told:?}");
    }

    #[test]
    fn short_affirmatives_said_quickly_acknowledge_us() {
        // A backchannel, however long it took.
        assert!(acknowledges("Mm-hmm.", 300));
        assert!(acknowledges("Yeah, got it.", 2_500));
        // Any number of affirmatives, "of course" and "go on" among them, said within 1.5 s.
        for said in ["Yeah, sure. Yeah, of course, yes.", "Of course.", "Go on.", "Okay, go on.", "Yes, yes, of course, sure."] {
            assert!(acknowledges(said, 1_500), "{said:?}");
        }
        // Not said quickly: more than an acknowledgement.
        assert!(!acknowledges("Yeah, sure. Yeah, of course, yes.", 2_000));
        assert!(!acknowledges("Of course.", 1_600));
        // Asking for something, or stopping us, never is.
        for said in ["Yes please", "Yeah, but wait", "Of course not", "Go on then, book it", "Okay stop", "Sure, Tuesday", "", "..."] {
            assert!(!acknowledges(said, 800), "{said:?}");
        }
        // The backchannel rule itself is as it was.
        assert!(!is_backchannel("Of course."));
        assert!(!is_backchannel("Go on."));
    }

    #[test]
    fn a_reply_cut_off_goes_on_from_the_first_line_not_heard_whole() {
        let line = |job, text: &str, item: Option<&str>, to_ms: Option<u64>| Line { job, text: text.into(), epoch: 0, greeting: false, goodbye: false, item: item.map(str::to_string), to_ms };
        let lines = vec![line(1, "One.", Some("out_1"), Some(1_500)), line(2, "Two.", Some("out_1"), Some(3_000)), line(3, "Three.", None, None)];
        // Cut in the second: said again from it, whole.
        assert_eq!(Resume::after(&lines, "out_1", 2_300), Some(Resume { item: "out_1".into(), from: 1, lines: vec!["Two.".into(), "Three.".into()] }));
        // At the first's very end: it was heard.
        assert_eq!(Resume::after(&lines, "out_1", 1_500).map(|r| r.from), Some(1));
        // In the first: all of it.
        assert_eq!(Resume::after(&lines, "out_1", 200).map(|r| r.lines.len()), Some(3));
        // A line still being spoken (its end not known yet) was not heard whole; one of another item neither.
        let speaking = vec![line(1, "One.", Some("out_1"), Some(1_500)), line(2, "Two.", Some("out_1"), None)];
        assert_eq!(Resume::after(&speaking, "out_1", 9_000).map(|r| r.lines), Some(vec!["Two.".to_string()]));
        assert_eq!(Resume::after(&lines, "out_2", 9_000).map(|r| r.from), Some(0));
        // All of it heard: nothing to go on with.
        assert_eq!(Resume::after(&lines[..2], "out_1", 3_000), None);
        // The greeting is never said again; a goodbye is never taken up.
        let mut greeting = lines.clone();
        greeting[0].greeting = true;
        assert_eq!(Resume::after(&greeting, "out_1", 200).map(|r| (r.from, r.lines)), Some((0, vec!["Two.".into(), "Three.".into()])));
        let mut goodbye = lines.clone();
        goodbye[2].goodbye = true;
        assert_eq!(Resume::after(&goodbye, "out_1", 200), None);
    }

    #[test]
    fn the_ledger_keeps_what_has_not_played_out() {
        let mut ledger = Ledger::default();
        let jobs: Vec<SpeakJob> = ["One.", "Two.", "Three."].iter().map(|t| SpeakJob::new(t.to_string(), 0, false, None)).collect();
        for job in &jobs {
            ledger.add(job);
        }
        ledger.began(jobs[0].job, "out_1");
        ledger.went(jobs[0].job, 1_500);
        ledger.began(jobs[1].job, "out_1");
        // A cut takes the reply's lines, as far as each went.
        let taken: Vec<_> = ledger.take(0).into_iter().map(|l| (l.text, l.item, l.to_ms)).collect();
        assert_eq!(taken, vec![("One.".into(), Some("out_1".into()), Some(1_500)), ("Two.".into(), Some("out_1".into()), None), ("Three.".into(), None, None)]);
        assert!(ledger.lines.is_empty());
        // An item that plays out takes its lines with it; the next reply's stay.
        let (four, five) = (SpeakJob::new("Four.".into(), 1, false, None), SpeakJob::new("Bye.".into(), 1, false, Some("tool_1".into())));
        ledger.add(&four);
        ledger.add(&five);
        ledger.began(four.job, "out_2");
        ledger.item_began = Some(("out_2".into(), Instant::now()));
        ledger.played("out_2");
        assert_eq!(ledger.lines.iter().map(|l| (l.text.as_str(), l.goodbye)).collect::<Vec<_>>(), vec![("Bye.", true)]);
        assert_eq!(ledger.item_began, None);
        assert_eq!(ledger.played_ms("out_2"), 0);
    }
}
