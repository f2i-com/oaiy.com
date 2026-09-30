//! Messages callers leave for the owner: what the receptionist takes when the owner cannot
//! be reached, in `<data>/messages/messages.json` (it holds numbers and words: made for its owner alone on Linux and macOS, and with the data
//! folder's own permissions on Windows, where nothing narrows it; see [`crate::secret_file`]).
//!
//! A message is `{id, at, callId, from, name, callback, message, urgency, wantsCallback, state,
//! seenAt, handledAt, handledBy}`. `from` is the number the phone said the call came from: this
//! desktop's own record of the call gives it, never the receptionist's model, so a caller
//! cannot have a message appear to come from someone else by saying so. `callback` is the number
//! they asked to be rung on (their own when they gave none). `state` is `new`, `seen` or `handled`.
//!
//! What is kept is bounded so a caller cannot fill the disk or the owner's screen:
//! 3 messages a call, 20 a number a day and 40 waiting at once, 600 characters each (control
//! characters removed), 2 000 in all. Callers who hide their number (or give none that is a
//! number) share ONE allowance, a small one, so hiding does not give every call a bucket of its
//! own: 6 a day and 100 kept between them. A handled message is let go after 90 days, and when
//! the store (or the hidden numbers' share of it) is full its oldest handled message makes room;
//! a message nobody has handled is never dropped to make room (the next is refused instead, the
//! receptionist says so, and the Messages page says the store is full: [`Store::notice`]).
//!
//! Reading, marking as seen or handled, and deleting are the dashboard's (`routes`); a message
//! is only ever made by the receptionist's `take_message` tool, through the call's own route
//! (`/api/voice/calls/:id/message`, `voice`).

pub mod routes;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use serde::{Deserialize, Serialize};

/// The messages' file, in `<data>/messages/`.
pub const FILE_NAME: &str = "messages.json";
/// The file's shape this desktop writes.
const VERSION: u64 = 1;
/// Messages taken on one call, at most.
pub const PER_CALL: usize = 3;
/// Messages one number may leave in a day, at most.
pub const PER_CALLER_DAY: usize = 20;
/// Messages one number may have waiting (new or seen) at once, at most.
pub const PER_NUMBER_WAITING: usize = 40;
/// Messages all hidden, withheld or unparseable numbers together may leave in a day, at most.
pub const PER_WITHHELD_DAY: usize = 6;
/// Messages all such callers together may have kept, at most (of any state).
pub const MAX_WITHHELD: usize = 100;
/// The longest message kept (characters).
pub const MAX_MESSAGE: usize = 600;
/// The longest name kept.
pub const MAX_NAME: usize = 80;
/// The longest callback number kept.
pub const MAX_CALLBACK: usize = 20;
/// The most messages held.
pub const MAX_STORED: usize = 2_000;
/// A handled message is let go after this many days.
pub const KEEP_HANDLED_DAYS: i64 = 90;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Urgency {
    #[default]
    Normal,
    Urgent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    #[default]
    New,
    Seen,
    Handled,
}

impl State {
    pub fn parse(s: &str) -> Option<State> {
        match s {
            "new" => Some(State::New),
            "seen" => Some(State::Seen),
            "handled" => Some(State::Handled),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    /// When it was taken (RFC 3339, UTC).
    pub at: String,
    pub call_id: String,
    /// The number the call came from, as the phone said ("" for a hidden number).
    pub from: String,
    /// The name the caller gave ("" when none).
    #[serde(default)]
    pub name: String,
    /// The number to ring back on: the one the caller gave, else the one they rang from ("" for a hidden number they gave none for).
    #[serde(default)]
    pub callback: String,
    pub message: String,
    #[serde(default)]
    pub urgency: Urgency,
    #[serde(default)]
    pub wants_callback: bool,
    #[serde(default)]
    pub state: State,
    #[serde(default)]
    pub seen_at: Option<String>,
    #[serde(default)]
    pub handled_at: Option<String>,
    #[serde(default)]
    pub handled_by: Option<String>,
}

/// What the receptionist gives to be recorded, with the number and call from this desktop's own record.
#[derive(Clone, Debug, Default)]
pub struct NewMessage {
    pub call_id: String,
    pub from: String,
    pub name: String,
    pub callback: String,
    pub message: String,
    pub urgent: bool,
    pub wants_callback: bool,
}

/// Why a request about messages was refused: an HTTP status, a code and words for the receptionist or the owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl Error {
    pub(crate) fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Default, Serialize, Deserialize)]
struct File {
    #[serde(default)]
    version: u64,
    #[serde(default)]
    messages: Vec<Message>,
}

/// The messages, in memory and on disk. Cloning shares them.
#[derive(Clone, Default)]
pub struct Store {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    path: Option<PathBuf>,
    messages: Vec<Message>,
    /// The file could not be read as messages and was put aside: nothing is lost by starting empty.
    quarantined: bool,
    /// What the Messages page says about the file that could not be used, in plain words.
    problem: Option<String>,
    /// The file could not be read and could not be put aside either: it is all that is left of what was in it, so
    /// nothing is written until the owner has dealt with it (every message offered is refused, and the receptionist
    /// says it could not be kept). A file that could not be read is read again (see `retry`), and it stops being protected when it can be.
    protected: bool,
    /// How long the file is waited for, and how the messages taken meanwhile are held.
    patience: crate::secret_file::Patience,
    /// Set while the file could not be read: when it is read again. The messages taken in its first moments are kept in memory and written when it is.
    retry: Option<crate::secret_file::Retry>,
}

impl Inner {
    /// Read the file into the store, waiting as `pauses` say for one that is busy. What the store holds already (messages taken while it could not be
    /// read) is merged with what is in it, and written if any was not there.
    fn load(&mut self, path: &Path, pauses: &[std::time::Duration]) {
        use crate::secret_file::{read_text_patiently, Retry, Text};
        let held = std::mem::take(&mut self.messages);
        let mut from_file = Vec::new();
        let why = match read_text_patiently(path, pauses) {
            Text::Missing => None,
            Text::Text(text) => match serde_json::from_str::<File>(&text) {
                Ok(file) if file.version <= VERSION => {
                    from_file = file.messages;
                    None
                }
                Ok(file) => Some(format!("it is from a newer OAIY (version {})", file.version)),
                Err(e) => Some(format!("it is not messages ({e})")),
            },
            Text::Undecodable(why) => Some(format!("it is not text ({why})")),
            Text::Unreadable(e) => {
                self.messages = held;
                let first = self.retry.is_none();
                match &mut self.retry {
                    Some(retry) => retry.failed(),
                    None => self.retry = Some(Retry::began(&self.patience)),
                }
                if first {
                    log::warn!("messages: {} could not be read ({e}); nothing will be written over it, and it is read again every so often", path.display());
                }
                self.quarantined = true;
                self.protected = true;
                self.problem = Some(format!(
                    "The file of saved messages could not be read ({e}): another program may have it open. OAIY is trying again; messages taken in the first few seconds are kept in memory and written when it can be, and after that no new message can be kept until it can be read. The file has not been changed."
                ));
                return;
            }
        };
        if let Some(why) = why {
            self.quarantined = true;
            match crate::secret_file::keep_aside(path) {
                Ok(aside) => {
                    let name = aside.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    log::warn!("messages: {} is not usable ({why}); it is kept as {name}", path.display());
                    self.problem = Some(format!("The file of saved messages could not be used ({why}). It is kept as {name} beside it, and new messages are kept in a new file."));
                }
                Err(e) => {
                    log::warn!("messages: {} is not usable ({why}) and could not be put aside ({e}); nothing will be written over it", path.display());
                    self.messages = held;
                    self.protected = true;
                    self.problem = Some(format!("The file of saved messages could not be used ({why}) and could not be put aside ({e}), so no new message can be kept until that is put right. It has not been changed."));
                    return;
                }
            }
        }
        // The file is read (or is not there, or was put aside): whatever was taken while it could not be is added to what it held, and written.
        let recovering = self.retry.take().is_some();
        let new_ones = held.iter().filter(|m| !from_file.iter().any(|f| f.id == m.id)).count();
        self.messages = from_file;
        let extra: Vec<Message> = held.into_iter().filter(|m| !self.messages.iter().any(|f| f.id == m.id)).collect();
        self.messages.extend(extra);
        if recovering {
            log::info!("messages: {} could be read again", path.display());
            if new_ones > 0 {
                if let Err(e) = Store::save(self) {
                    log::warn!("messages: the {new_ones} taken while the file could not be read could not be written: {}", e.message);
                }
            }
        }
    }

    /// The file could not be read at the last try, and it is time to read it again.
    fn recover(&mut self) {
        let Some(path) = self.path.clone() else { return };
        self.protected = false;
        self.quarantined = false;
        self.problem = None;
        self.load(&path, &[]);
    }

    /// Messages taken now are kept in memory, to be written when the file can be read: it is busy, and it was only just found so.
    fn holds_in_memory(&self) -> bool {
        self.protected && self.retry.as_ref().is_some_and(|r| r.in_window())
    }
}

/// `text` as it may be shown and kept: control and direction-changing characters removed, tabs and
/// line breaks made spaces, runs of spaces one, trimmed, and at most `max` characters.
pub fn clean(text: &str, max: usize) -> String {
    let mut out = String::new();
    let mut gap = false;
    for c in text.chars() {
        let hidden = matches!(c, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{2069}' | '\u{feff}');
        if c.is_whitespace() || matches!(c, '\t' | '\n' | '\r') {
            gap = true;
        } else if c.is_control() || hidden {
            continue;
        } else {
            if gap && !out.is_empty() {
                out.push(' ');
            }
            gap = false;
            out.push(c);
        }
    }
    out.chars().take(max).collect::<String>().trim_end().to_string()
}

/// A callback number as it is kept: digits and a leading `+` only, at most [`MAX_CALLBACK`] characters; "" when it has no digits.
pub fn clean_number(text: &str) -> String {
    let text = text.trim();
    let plus = text.starts_with('+');
    let digits: String = text.chars().filter(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return String::new();
    }
    let mut out = if plus { format!("+{digits}") } else { digits };
    out.truncate(MAX_CALLBACK);
    out
}

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

impl Store {
    /// The messages kept in `<data>/messages/messages.json` (UTF-8, or UTF-16 with its byte order mark, as another program
    /// may have saved it). A file that is not messages, or is not text, is put aside as `messages.json.corrupt` (or
    /// `.corrupt.1`, and so on: never written over, and no earlier one replaced) and the store starts empty. A file that
    /// cannot be read at all is not touched, and nothing is written over it: see `protected`.
    pub fn open(data_dir: &Path) -> Store {
        Store::open_patiently(data_dir, crate::secret_file::Patience::default())
    }

    /// [`Store::open`], waiting for a file that is busy as `patience` says: it is read again after each of its short pauses, and if it is still busy the
    /// store opens without it (the owner is told on the Messages page, and it is logged), is read again every so often, and merges what was taken in memory
    /// meanwhile when it can be. A transient lock at start-up (an antivirus scan, an indexer) does not turn message keeping off until the next start.
    pub fn open_patiently(data_dir: &Path, patience: crate::secret_file::Patience) -> Store {
        let path = data_dir.join("messages").join(FILE_NAME);
        let mut inner = Inner { path: Some(path.clone()), patience: patience.clone(), ..Inner::default() };
        inner.load(&path, &patience.start);
        Store { inner: Arc::new(Mutex::new(inner)) }
    }

    /// Messages held in memory only (tests).
    pub fn in_memory() -> Store {
        Store::default()
    }

    /// The store, read again first if its file was busy and it is time to try (each use is a chance to).
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.retry.as_ref().is_some_and(|r| r.due()) {
            inner.recover();
        }
        inner
    }

    fn save(inner: &Inner) -> Result<(), Error> {
        let Some(path) = &inner.path else { return Ok(()) };
        if inner.protected {
            return Err(Error::new(500, "save_failed", "the message could not be saved: the file of saved messages could not be read or put aside, and is not written over"));
        }
        let body =serde_json::to_string_pretty(&File { version: VERSION, messages: inner.messages.clone() }).map_err(|e| Error::new(500, "save_failed", e.to_string()))?;
        crate::secret_file::write(path, body).map_err(|e| Error::new(500, "save_failed", format!("the message could not be saved: {e}")))
    }

    /// Keep a message the receptionist took. The limits, and the words cleaned, are here: the same
    /// message said again on the same call is the one already kept.
    pub fn add(&self, new: NewMessage) -> Result<Message, Error> {
        self.add_at(new, now())
    }

    /// [`Store::add`] as of `when` (the limits are by the clock: a test sets it).
    pub(crate) fn add_at(&self, new: NewMessage, when: chrono::DateTime<chrono::Utc>) -> Result<Message, Error> {
        let text = clean(&new.message, MAX_MESSAGE);
        if text.is_empty() {
            return Err(Error::new(400, "empty_message", "there is no message to keep: ask what the caller wants the owner to know"));
        }
        let key = crate::voice::contacts::key(&new.from);
        let mut inner = self.lock();
        // Who counts as one caller: their number, or, for every hidden or unparseable number, all of them together.
        let same_caller = |m: &Message| match &key {
            Some(k) => crate::voice::contacts::key(&m.from).as_deref() == Some(k.as_str()),
            None => crate::voice::contacts::key(&m.from).is_none(),
        };
        if let Some(existing) = inner.messages.iter().find(|m| m.call_id == new.call_id && m.message == text) {
            return Ok(existing.clone());
        }
        if inner.messages.iter().filter(|m| m.call_id == new.call_id).count() >= PER_CALL {
            return Err(Error::new(429, "call_limit", format!("{PER_CALL} messages have been taken on this call: no more can be kept")));
        }
        let today = inner.messages.iter().filter(|m| same_caller(m) && chrono::DateTime::parse_from_rfc3339(&m.at).map(|at| when.signed_duration_since(at).num_hours() < 24).unwrap_or(false)).count();
        if key.is_none() {
            if today >= PER_WITHHELD_DAY {
                return Err(Error::new(429, "withheld_limit", "callers who hide their number have left as many messages today as are kept: ask them to ring back with their number showing"));
            }
        } else if today >= PER_CALLER_DAY {
            return Err(Error::new(429, "caller_limit", "this number has left as many messages today as are kept"));
        }
        if key.is_some() && inner.messages.iter().filter(|m| same_caller(m) && m.state != State::Handled).count() >= PER_NUMBER_WAITING {
            return Err(Error::new(429, "caller_waiting", format!("{PER_NUMBER_WAITING} messages from this number are waiting for the owner: no more can be kept until some are handled")));
        }
        // Old handled messages go; if it is still full, the oldest handled one makes room, and an unhandled one never does.
        let cutoff = when - chrono::Duration::days(KEEP_HANDLED_DAYS);
        inner.messages.retain(|m| m.state != State::Handled || m.handled_at.as_deref().and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok()).is_none_or(|t| t >= cutoff));
        // Hidden numbers share a small part of the store: at its limit their oldest handled message makes room, and if all
        // of theirs are waiting, this one is refused (and the Messages page says so).
        if key.is_none() && inner.messages.iter().filter(|m| crate::voice::contacts::key(&m.from).is_none()).count() >= MAX_WITHHELD {
            let oldest = inner.messages.iter().enumerate().filter(|(_, m)| m.state == State::Handled && crate::voice::contacts::key(&m.from).is_none()).min_by(|a, b| a.1.at.cmp(&b.1.at)).map(|(i, _)| i);
            match oldest {
                Some(i) => {
                    inner.messages.remove(i);
                }
                None => return Err(Error::new(507, "withheld_full", "messages from callers who hid their number are waiting for the owner: no more of those can be kept until some are handled")),
            }
        }
        if inner.messages.len() >= MAX_STORED {
            let oldest = inner.messages.iter().enumerate().filter(|(_, m)| m.state == State::Handled).min_by(|a, b| a.1.at.cmp(&b.1.at)).map(|(i, _)| i);
            match oldest {
                Some(i) => {
                    inner.messages.remove(i);
                }
                None => return Err(Error::new(507, "store_full", "the owner has too many messages waiting to keep another")),
            }
        }
        let callback = clean_number(&new.callback);
        let message = Message {
            id: format!("msg_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]),
            at: when.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            call_id: new.call_id.clone(),
            from: clean(&new.from, 40),
            name: clean(&new.name, MAX_NAME),
            callback: if callback.is_empty() { clean(&new.from, 40) } else { callback },
            message: text,
            urgency: if new.urgent { Urgency::Urgent } else { Urgency::Normal },
            wants_callback: new.wants_callback,
            state: State::New,
            seen_at: None,
            handled_at: None,
            handled_by: None,
        };
        inner.messages.push(message.clone());
        if let Err(e) = Self::save(&inner) {
            // The file is busy and was only just found so: the message is kept in memory and written when it can be read (the owner is told, on the
            // Messages page, that it could not be yet). After those first moments a message that cannot be written is refused, as it always was.
            if inner.holds_in_memory() {
                return Ok(message);
            }
            inner.messages.pop();
            return Err(e);
        }
        Ok(message)
    }

    /// Every message, newest first, of `state` (all when None), and those whose name, number or words hold `q`.
    pub fn list(&self, state: Option<State>, q: &str) -> Vec<Message> {
        let q = q.trim().to_lowercase();
        let digits: String = q.chars().filter(char::is_ascii_digit).collect();
        // A number is found written any way: by its last nine digits, as contacts are.
        let needle = &digits[digits.len().saturating_sub(9)..];
        let mut found: Vec<Message> = self
            .lock()
            .messages
            .iter()
            .filter(|m| state.is_none_or(|s| m.state == s))
            .filter(|m| {
                q.is_empty()
                    || m.name.to_lowercase().contains(&q)
                    || m.message.to_lowercase().contains(&q)
                    || (digits.len() >= 3 && [&m.from, &m.callback].iter().any(|n| n.chars().filter(char::is_ascii_digit).collect::<String>().contains(needle)))
            })
            .cloned()
            .collect();
        found.sort_by(|a, b| b.at.cmp(&a.at).then_with(|| b.id.cmp(&a.id)));
        found
    }

    /// A plain word for the Messages page when the file of messages could not be used (and where it is kept), or when new
    /// messages are being refused because the owner has not handled the ones waiting (none when all is well).
    pub fn notice(&self) -> Option<String> {
        let inner = self.lock();
        if let Some(problem) = &inner.problem {
            return Some(problem.clone());
        }
        let waiting = inner.messages.iter().filter(|m| m.state != State::Handled).count();
        let hidden = inner.messages.iter().filter(|m| crate::voice::contacts::key(&m.from).is_none());
        let (hidden_all, hidden_waiting) = hidden.fold((0, 0), |(all, waiting), m| (all + 1, waiting + usize::from(m.state != State::Handled)));
        if waiting >= MAX_STORED {
            Some(format!("{waiting} messages are waiting and no more can be kept: new messages are refused until you mark some as handled or delete them."))
        } else if hidden_all >= MAX_WITHHELD && hidden_waiting >= MAX_WITHHELD {
            Some(format!("{hidden_waiting} messages from callers who hid their number are waiting: new ones from hidden numbers are refused until you mark some as handled or delete them."))
        } else {
            None
        }
    }

    /// How many messages are new.
    pub fn unread(&self) -> usize {
        self.lock().messages.iter().filter(|m| m.state == State::New).count()
    }

    pub fn get(&self, id: &str) -> Option<Message> {
        self.lock().messages.iter().find(|m| m.id == id).cloned()
    }

    /// Mark a message `seen`, `handled` (by `by`) or `new` again.
    pub fn set_state(&self, id: &str, state: State, by: &str) -> Result<Message, Error> {
        let mut inner = self.lock();
        let at = now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let Some(i) = inner.messages.iter().position(|m| m.id == id) else {
            return Err(Error::new(404, "no_message", format!("no message {id:?}")));
        };
        let before = inner.messages[i].clone();
        {
            let m = &mut inner.messages[i];
            m.state = state;
            match state {
                State::New => {
                    m.seen_at = None;
                    m.handled_at = None;
                    m.handled_by = None;
                }
                State::Seen => {
                    m.seen_at.get_or_insert(at.clone());
                    m.handled_at = None;
                    m.handled_by = None;
                }
                State::Handled => {
                    m.seen_at.get_or_insert(at.clone());
                    m.handled_at = Some(at);
                    m.handled_by = Some(clean(by, 40)).filter(|b| !b.is_empty());
                }
            }
        }
        if let Err(e) = Self::save(&inner) {
            inner.messages[i] = before;
            return Err(e);
        }
        Ok(inner.messages[i].clone())
    }

    /// Forget a message.
    pub fn remove(&self, id: &str) -> Result<(), Error> {
        let mut inner = self.lock();
        let Some(i) = inner.messages.iter().position(|m| m.id == id) else {
            return Err(Error::new(404, "no_message", format!("no message {id:?}")));
        };
        let removed = inner.messages.remove(i);
        if let Err(e) = Self::save(&inner) {
            inner.messages.insert(i, removed);
            return Err(e);
        }
        Ok(())
    }

    /// Whether the file could not be read at the start (its bytes are kept beside it).
    pub fn was_quarantined(&self) -> bool {
        self.lock().quarantined
    }
}

// ---- telling the owner ----------------------------------------------------------

/// What tells the owner a message arrived: on the GUI, a native notification (a stand-in in tests, nothing on the headless server).
pub trait MessageNotifier: Send + Sync {
    /// Tell the owner. Whether a person could have been told (a notification was raised).
    fn message_taken(&self, message: &Message) -> bool;
}

static NOTIFIER: RwLock<Option<Arc<dyn MessageNotifier>>> = RwLock::new(None);

/// Use `notifier` to tell the owner of a message from now on.
pub fn set_notifier(notifier: Option<Arc<dyn MessageNotifier>>) {
    if let Ok(mut n) = NOTIFIER.write() {
        *n = notifier;
    }
}

/// Tell the owner `message` was taken: whether anything could tell them.
pub fn notify(message: &Message) -> bool {
    NOTIFIER.read().ok().and_then(|n| n.clone()).is_some_and(|n| n.message_taken(message))
}

static SHARED: OnceLock<Store> = OnceLock::new();

/// Open this desktop's messages in its data folder.
pub fn init(data_dir: &Path) {
    let _ = SHARED.set(Store::open(data_dir));
}

/// This desktop's messages (empty and unsaved before [`init`]).
pub fn shared() -> Store {
    SHARED.get().cloned().unwrap_or_default()
}

#[cfg(test)]
mod tests;
