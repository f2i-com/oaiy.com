//! Contacts: the people who ring and text.
//!
//! Kept in `<data>/callers.json` (where the names callers were greeted by were
//! kept before), by each number's key: its last nine digits, so +61 491 570
//! 006 and 0491 570 006 are one person. The file is
//! `{"version": 2, "contacts": {key: contact}}`, and a contact is
//!
//! ```json
//! {"key": "491570006", "number": "+61491570006", "name": "Lance", "nameBy": "owner",
//!  "notes": "…", "facts": [{"text": "…", "at": "…", "by": "agent"}],
//!  "createdAt": "…", "updatedAt": "…"}
//! ```
//!
//! - `number` is the number last seen for them (a call's, or the one the
//!   person gave); empty for a contact from the older file, which kept only
//!   the key.
//! - `name` may be one word ("Lance"). `nameBy` says who named them: `owner`
//!   (the person: Contacts, an import, their Agent) or `agent` (the
//!   receptionist, from a call); `null` while they have no name. The
//!   receptionist never renames someone the person named.
//! - `notes` are the person's own, for the receptionist to read on every call
//!   and text with them; `facts` are what the receptionist remembered.
//! - A hidden or withheld caller never becomes a contact.
//!
//! A number is seen on every call ([`saw`]), so contacts appear as people
//! ring. The older file, `{"491570006": "Lance"}` (names the receptionist
//! learned), is upgraded in place the first time it is read: its bytes are
//! kept once as `callers.json.bak`, then the new file is written whole (a
//! temporary file, then a rename). A file that cannot be read is never
//! written over.

pub mod csv;
pub mod routes;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The contacts' file, in the data folder.
pub const FILE_NAME: &str = "callers.json";
/// The file's shape this desktop writes.
const VERSION: u64 = 2;
/// The longest name kept (a longer one is cut).
pub const MAX_NAME: usize = 80;
/// The most notes a contact has (more is refused, never cut).
pub const MAX_NOTES: usize = 2000;
/// The most facts a contact keeps.
pub const MAX_FACTS: usize = 50;
/// The longest fact (more is refused).
pub const MAX_FACT: usize = 300;
/// The longest number kept as it was given.
const MAX_NUMBER: usize = 40;

/// Who named a contact, or wrote a fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum By {
    Owner,
    Agent,
}

impl By {
    pub fn parse(s: &str) -> Option<By> {
        match s.trim().to_ascii_lowercase().as_str() {
            "owner" => Some(By::Owner),
            "agent" => Some(By::Agent),
            _ => None,
        }
    }
}

/// Something the receptionist remembered about a contact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub text: String,
    pub at: String,
    pub by: By,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Contact {
    pub key: String,
    #[serde(default)]
    pub number: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub name_by: Option<By>,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub facts: Vec<Fact>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
}

impl Contact {
    fn new(key: &str, now: &str) -> Contact {
        Contact {
            key: key.to_string(),
            number: String::new(),
            name: String::new(),
            name_by: None,
            notes: String::new(),
            facts: Vec::new(),
            created_at: now.to_string(),
            updated_at: now.to_string(),
        }
    }

    /// Whether the person named them (the receptionist's name never replaces theirs).
    pub fn named_by_owner(&self) -> bool {
        self.name_by == Some(By::Owner) && !self.name.is_empty()
    }
}

/// The file: every contact, by key.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Book {
    #[serde(default)]
    version: u64,
    #[serde(default)]
    contacts: BTreeMap<String, Contact>,
}

impl Book {
    fn empty() -> Book {
        Book { version: VERSION, contacts: BTreeMap::new() }
    }
}

/// Why a request about contacts was refused: an HTTP status, a code and words for a person.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl Error {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Error {
        Error { status, code, message: message.into() }
    }

    fn no_contact(key: &str) -> Error {
        Error::new(404, "no_contact", format!("there is no contact whose number ends {key}"))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/// Why a number cannot be a contact's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotANumber {
    /// No number at all.
    Missing,
    /// A hidden or withheld caller ("Private", "Unknown", Android's -1...).
    Hidden,
    /// Too few digits to be a phone's (fewer than 8).
    TooShort,
}

impl NotANumber {
    pub fn code(self) -> &'static str {
        match self {
            NotANumber::Missing => "no_number",
            NotANumber::Hidden => "hidden",
            NotANumber::TooShort => "not_a_phone_number",
        }
    }

    pub fn words(self) -> &'static str {
        match self {
            NotANumber::Missing => "no number",
            NotANumber::Hidden => "a hidden or withheld number",
            NotANumber::TooShort => "not a phone number",
        }
    }
}

/// What a phone shows for a caller who hides their number.
const HIDDEN_WORDS: [&str; 10] = ["unknown", "private", "withheld", "anonymous", "restricted", "blocked", "unavailable", "hidden", "no caller", "payphone"];

/// A number's key (its last nine digits), or why it has none.
pub fn check_number(number: &str) -> Result<String, NotANumber> {
    let text = number.trim();
    if text.is_empty() {
        return Err(NotANumber::Missing);
    }
    let lower = text.to_lowercase();
    let digits: Vec<char> = text.chars().filter(char::is_ascii_digit).collect();
    let all_zero = !digits.is_empty() && digits.iter().all(|d| *d == '0');
    if HIDDEN_WORDS.iter().any(|w| lower.contains(w)) || matches!(text, "-1" | "-2" | "-3") || all_zero {
        return Err(NotANumber::Hidden);
    }
    if digits.is_empty() {
        return Err(NotANumber::Missing);
    }
    if digits.len() < 8 {
        return Err(NotANumber::TooShort);
    }
    Ok(digits[digits.len().saturating_sub(9)..].iter().collect())
}

/// A number's key: its last nine digits; none for a hidden caller or one with too few digits.
pub fn key(number: &str) -> Option<String> {
    check_number(number).ok()
}

fn key_of(number: &str) -> Result<String, Error> {
    check_number(number).map_err(|why| match why {
        NotANumber::Hidden => Error::new(400, why.code(), "a hidden or withheld number never becomes a contact"),
        _ => Error::new(400, why.code(), format!("{:?} is {}: give a phone number with at least 8 digits", number.trim(), why.words())),
    })
}

/// A country whose numbers may be written the local way (`0491 570 006`).
#[derive(Debug, PartialEq, Eq)]
pub struct Country {
    pub code: &'static str,
    pub name: &'static str,
    /// Its calling code, without the plus.
    calling: &'static str,
    /// Whether a local number starts with a 0 that is dropped after the calling code.
    trunk: bool,
    /// How many digits its numbers have after the calling code.
    lengths: &'static [usize],
}

/// The countries an import's local numbers may be from.
pub const COUNTRIES: &[Country] = &[
    Country { code: "AU", name: "Australia", calling: "61", trunk: true, lengths: &[9] },
    Country { code: "NZ", name: "New Zealand", calling: "64", trunk: true, lengths: &[8, 9, 10] },
    Country { code: "GB", name: "United Kingdom", calling: "44", trunk: true, lengths: &[10] },
    Country { code: "IE", name: "Ireland", calling: "353", trunk: true, lengths: &[9] },
    Country { code: "US", name: "United States", calling: "1", trunk: false, lengths: &[10] },
    Country { code: "CA", name: "Canada", calling: "1", trunk: false, lengths: &[10] },
    Country { code: "ZA", name: "South Africa", calling: "27", trunk: true, lengths: &[9] },
    Country { code: "IN", name: "India", calling: "91", trunk: true, lengths: &[10] },
    Country { code: "SG", name: "Singapore", calling: "65", trunk: false, lengths: &[8] },
];

/// The country assumed when none is given (the app keeps its own setting and passes it).
pub const DEFAULT_COUNTRY: &str = "AU";

pub fn country(code: &str) -> Option<&'static Country> {
    let code = code.trim().to_ascii_uppercase();
    let code = if code == "UK" { "GB".to_string() } else { code };
    COUNTRIES.iter().find(|c| c.code == code)
}

/// A number as dialled from anywhere (`+61491570006`), reading one written
/// the local way (`0491 570 006`) as `country`'s; as it was given when it
/// cannot tell.
pub fn international(number: &str, country: &Country) -> String {
    let text = number.trim();
    let digits: String = text.chars().filter(char::is_ascii_digit).collect();
    if text.starts_with('+') {
        return format!("+{digits}");
    }
    for exit in ["00", "011"] {
        if let Some(rest) = digits.strip_prefix(exit) {
            if rest.len() >= 8 && !(exit == "00" && country.trunk && country.lengths.contains(&(digits.len() - 1))) {
                return format!("+{rest}");
            }
        }
    }
    if country.trunk {
        if let Some(rest) = digits.strip_prefix('0') {
            if country.lengths.contains(&rest.len()) {
                return format!("+{}{rest}", country.calling);
            }
        }
    }
    if country.lengths.contains(&digits.len()) {
        return format!("+{}{digits}", country.calling);
    }
    if let Some(rest) = digits.strip_prefix(country.calling) {
        if country.lengths.contains(&rest.len()) {
            return format!("+{digits}");
        }
    }
    given_number(text)
}

/// A number kept as it was given (trimmed, and not too long).
fn given_number(number: &str) -> String {
    number.trim().chars().take(MAX_NUMBER).collect()
}

// ---------------------------------------------------------------------------
// What is kept, cleaned
// ---------------------------------------------------------------------------

/// A name as it is kept: its words, single-spaced, at most [`MAX_NAME`]
/// characters. Nothing else about it is changed: "Lance" stays "Lance", and
/// "lance" stays "lance".
pub fn clean_name(name: &str) -> String {
    name.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_NAME).collect()
}

/// Notes as they are kept: one kind of line break, trimmed; refused past [`MAX_NOTES`].
pub fn clean_notes(notes: &str) -> Result<String, Error> {
    let notes = notes.replace("\r\n", "\n").replace('\r', "\n");
    let notes = notes.trim();
    let n = notes.chars().count();
    if n > MAX_NOTES {
        return Err(Error::new(400, "too_long", format!("notes are at most {MAX_NOTES} characters (these are {n})")));
    }
    Ok(notes.to_string())
}

/// A fact as it is kept: one line, single-spaced; refused when empty or past [`MAX_FACT`].
pub fn clean_fact(text: &str) -> Result<String, Error> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let n = text.chars().count();
    if n == 0 {
        return Err(Error::new(400, "empty_fact", "a fact needs some words"));
    }
    if n > MAX_FACT {
        return Err(Error::new(400, "too_long", format!("a fact is at most {MAX_FACT} characters (this is {n}): say it shorter")));
    }
    Ok(text)
}

/// Add a fact to `c`, unless it is there already (in any case); a full list
/// lets its oldest of the receptionist's go. Answers whether it was added,
/// and the fact let go.
fn remember_fact(c: &mut Contact, text: &str, by: By, now: &str) -> Result<(bool, Option<Fact>), Error> {
    let lower = text.to_lowercase();
    if c.facts.iter().any(|f| f.text.to_lowercase() == lower) {
        return Ok((false, None));
    }
    let mut dropped = None;
    while c.facts.len() >= MAX_FACTS {
        match c.facts.iter().position(|f| f.by == By::Agent) {
            Some(i) => dropped = Some(c.facts.remove(i)),
            None => {
                return Err(Error::new(409, "facts_full", format!("this contact has {MAX_FACTS} facts the person wrote: forget one first")));
            }
        }
    }
    c.facts.push(Fact { text: text.to_string(), at: now.to_string(), by });
    Ok((true, dropped))
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Whether `q` finds `c`: in its name, notes, facts or number, in any case;
/// a number typed any way (its digits, with or without the leading 0).
fn matches(c: &Contact, q: &str) -> bool {
    let q = q.trim();
    if q.is_empty() {
        return true;
    }
    let lower = q.to_lowercase();
    let words = [c.name.as_str(), c.notes.as_str(), c.number.as_str()];
    if words.iter().any(|w| w.to_lowercase().contains(&lower)) || c.facts.iter().any(|f| f.text.to_lowercase().contains(&lower)) {
        return true;
    }
    if !q.chars().all(|ch| ch.is_ascii_digit() || " +-().".contains(ch)) {
        return false;
    }
    let digits: String = q.chars().filter(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return false;
    }
    let number: String = c.number.chars().filter(char::is_ascii_digit).collect();
    let local = digits.trim_start_matches('0');
    [digits.as_str(), local].iter().filter(|d| !d.is_empty()).any(|d| number.contains(d) || c.key.contains(d))
}

/// By name, in any case; those with no name after, by number.
fn sort(list: &mut [Contact]) {
    list.sort_by(|a, b| {
        let (an, bn) = (a.name.to_lowercase(), b.name.to_lowercase());
        (an.is_empty(), an, &a.key).cmp(&(bn.is_empty(), bn, &b.key))
    });
}

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

/// One change at a time: the file is read, changed and written whole.
static CHANGING: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    CHANGING.lock().unwrap_or_else(|e| e.into_inner())
}

fn damaged(file: &Path, why: impl std::fmt::Display) -> Error {
    Error::new(
        500,
        "contacts_unreadable",
        format!("{} could not be read ({why}): it is left as it is, and nothing is saved over it", file.display()),
    )
}

/// The book in `file`: none yet is empty; the older names-only file is upgraded (and kept as `.bak`).
fn open(file: &Path) -> Result<Book, Error> {
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Book::empty()),
        Err(e) => return Err(damaged(file, e)),
    };
    let text = String::from_utf8_lossy(&bytes);
    let text = text.trim_start_matches('\u{feff}');
    if text.trim().is_empty() {
        return Ok(Book::empty());
    }
    let value: Value = serde_json::from_str(text).map_err(|e| damaged(file, e))?;
    let Value::Object(map) = &value else {
        return Err(damaged(file, "it is not a JSON object"));
    };
    if map.contains_key("version") || map.contains_key("contacts") {
        let mut book: Book = serde_json::from_value(value).map_err(|e| damaged(file, e))?;
        // A contact's key is its place in the file.
        for (k, c) in book.contacts.iter_mut() {
            c.key.clone_from(k);
        }
        return Ok(book);
    }
    if map.values().all(Value::is_string) {
        let book = upgraded(map, &now());
        keep_backup(file, &bytes)?;
        save(file, &book)?;
        log::info!("contacts: {} upgraded to contacts ({} named), the older file kept as .bak", file.display(), book.contacts.len());
        return Ok(book);
    }
    Err(damaged(file, "it is neither contacts nor the older names"))
}

/// The older file's names (`{"491570006": "Lance"}`), as contacts. The
/// receptionist wrote every one of them (`PUT /api/voice/callers`), so they
/// are its: the person makes a name theirs by saving it in Contacts.
fn upgraded(names: &serde_json::Map<String, Value>, now: &str) -> Book {
    let mut book = Book::empty();
    for (k, v) in names {
        let (Ok(key), Some(name)) = (check_number(k), v.as_str().map(clean_name)) else {
            log::warn!("contacts: the older file's entry {k:?} is not a caller's number, and is left out");
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let mut c = Contact::new(&key, now);
        c.name = name;
        c.name_by = Some(By::Agent);
        book.contacts.insert(key, c);
    }
    book
}

/// `callers.json.bak`: the older file's bytes, kept once (never written over).
fn keep_backup(file: &Path, bytes: &[u8]) -> Result<(), Error> {
    let name = file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| FILE_NAME.to_string());
    let bak = file.with_file_name(format!("{name}.bak"));
    if bak.exists() {
        return Ok(());
    }
    std::fs::write(&bak, bytes).map_err(|e| Error::new(500, "save_failed", format!("{} could not be kept before the upgrade: {e}", bak.display())))
}

fn save(file: &Path, book: &Book) -> Result<(), Error> {
    let value = serde_json::to_value(Book { version: VERSION, contacts: book.contacts.clone() }).map_err(|e| Error::new(500, "save_failed", e.to_string()))?;
    crate::control::write_atomic(file, &value).map_err(|e| Error::new(500, "save_failed", e))
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// The contacts in one file. Cheap to clone: it is the file's path.
#[derive(Clone, Debug, Default)]
pub struct Store {
    file: Option<PathBuf>,
}

/// A change's answer, and whether it saved anything.
type Changed<T> = Result<(T, bool), Error>;

/// Something remembered: the contact, whether the fact was new, and the fact a full list let go.
#[derive(Debug, Serialize)]
pub struct Remembered {
    pub contact: Contact,
    pub added: bool,
    pub dropped: Option<Fact>,
}

/// What `PUT /api/contacts/:key` changes: only the fields given.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Change {
    pub name: Option<String>,
    pub notes: Option<String>,
    pub number: Option<String>,
}

impl Store {
    pub fn at(file: PathBuf) -> Store {
        Store { file: Some(file) }
    }

    fn file(&self) -> Result<&Path, Error> {
        self.file.as_deref().ok_or_else(|| Error::new(503, "contacts_unavailable", "the contacts are not open yet"))
    }

    /// Read the book (upgrading the older file), one at a time with the changes.
    fn read<T>(&self, f: impl FnOnce(&Book) -> Result<T, Error>) -> Result<T, Error> {
        let file = self.file()?;
        let _one = lock();
        f(&open(file)?)
    }

    /// Change the book, and save it whole when `f` says it changed.
    fn change<T>(&self, f: impl FnOnce(&mut Book, &str) -> Changed<T>) -> Result<T, Error> {
        let file = self.file()?;
        let _one = lock();
        let mut book = open(file)?;
        if book.version > VERSION {
            return Err(Error::new(409, "contacts_newer", format!("{} was written by a newer OAIY: update this one to change contacts", file.display())));
        }
        let (out, changed) = f(&mut book, &now())?;
        if changed {
            save(file, &book)?;
        }
        Ok(out)
    }

    /// Every contact `q` finds (all for an empty one), by name; and how many there are in all.
    pub fn list(&self, q: &str) -> Result<(Vec<Contact>, usize), Error> {
        self.read(|book| {
            let mut found: Vec<Contact> = book.contacts.values().filter(|c| matches(c, q)).cloned().collect();
            sort(&mut found);
            Ok((found, book.contacts.len()))
        })
    }

    /// Every contact, by name.
    pub fn all(&self) -> Result<Vec<Contact>, Error> {
        self.list("").map(|(all, _)| all)
    }

    /// The contact for a number written any way.
    pub fn get(&self, number: &str) -> Result<Contact, Error> {
        let key = key_of(number)?;
        self.read(|book| book.contacts.get(&key).cloned().ok_or_else(|| Error::no_contact(&key)))
    }

    /// The name a caller is greeted by.
    pub fn name_of(&self, number: &str) -> Option<String> {
        let key = key(number)?;
        self.read(|book| Ok(book.contacts.get(&key).map(|c| c.name.clone()).filter(|n| !n.is_empty()))).ok().flatten()
    }

    /// A number seen on a call: its contact, made if there is none, with this
    /// as its number. Nothing for a hidden caller. Saved only when it changed.
    pub fn saw(&self, number: &str) -> Result<Option<Contact>, Error> {
        let Some(key) = key(number) else { return Ok(None) };
        let given = given_number(number);
        self.change(|book, now| {
            let new = !book.contacts.contains_key(&key);
            let c = book.contacts.entry(key.clone()).or_insert_with(|| Contact::new(&key, now));
            let renumbered = c.number != given;
            if renumbered {
                c.number.clone_from(&given);
                c.updated_at = now.to_string();
            }
            Ok((Some(c.clone()), new || renumbered))
        })
    }

    /// The receptionist's name for a caller (`PUT /api/voice/callers`): kept
    /// unless the person named them, when theirs stays; an empty one clears
    /// only the receptionist's. The contact (notes, facts) is never removed.
    /// Answers with the name kept.
    pub fn remember_name(&self, number: &str, name: &str) -> Result<String, Error> {
        let key = key_of(number)?;
        let name = clean_name(name);
        let given = given_number(number);
        self.change(|book, now| {
            let exists = book.contacts.contains_key(&key);
            if !exists && name.is_empty() {
                return Ok((String::new(), false));
            }
            let c = book.contacts.entry(key.clone()).or_insert_with(|| Contact::new(&key, now));
            let mut changed = !exists;
            // Their whole number, when it is more than the key.
            let fuller = given.chars().filter(char::is_ascii_digit).count() > key.len();
            if fuller && c.number != given {
                c.number.clone_from(&given);
                changed = true;
            }
            if !c.named_by_owner() {
                let by = (!name.is_empty()).then_some(By::Agent);
                if c.name != name || c.name_by != by {
                    c.name.clone_from(&name);
                    c.name_by = by;
                    changed = true;
                }
            }
            if changed {
                c.updated_at = now.to_string();
            }
            Ok((c.name.clone(), changed))
        })
    }

    /// The person's change to a contact (a name set here is theirs), made if there is none.
    pub fn set(&self, number: &str, change: Change) -> Result<Contact, Error> {
        let key = key_of(number)?;
        let notes = change.notes.as_deref().map(clean_notes).transpose()?;
        let name = change.name.as_deref().map(clean_name);
        let given = match change.number.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
            Some(n) if key_of(n)? != key => {
                return Err(Error::new(400, "other_number", format!("{n:?} is another contact's number: its last nine digits are not {key}")));
            }
            Some(n) => Some(given_number(n)),
            None => None,
        };
        self.change(|book, now| {
            let new = !book.contacts.contains_key(&key);
            let c = book.contacts.entry(key.clone()).or_insert_with(|| Contact::new(&key, now));
            let mut changed = new;
            if let Some(name) = name {
                let by = (!name.is_empty()).then_some(By::Owner);
                if c.name != name || c.name_by != by {
                    c.name = name;
                    c.name_by = by;
                    changed = true;
                }
            }
            if let Some(notes) = notes {
                if c.notes != notes {
                    c.notes = notes;
                    changed = true;
                }
            }
            if let Some(n) = given {
                if c.number != n {
                    c.number = n;
                    changed = true;
                }
            }
            if changed {
                c.updated_at = now.to_string();
            }
            Ok((c.clone(), changed))
        })
    }

    /// Forget a contact: their name, notes and facts.
    pub fn remove(&self, number: &str) -> Result<Contact, Error> {
        let key = key_of(number)?;
        self.change(|book, _| book.contacts.remove(&key).map(|c| (c, true)).ok_or_else(|| Error::no_contact(&key)))
    }

    /// Remember a fact about a contact (made if there is none).
    pub fn add_fact(&self, number: &str, text: &str, by: By) -> Result<Remembered, Error> {
        let key = key_of(number)?;
        let text = clean_fact(text)?;
        self.change(|book, now| {
            let new = !book.contacts.contains_key(&key);
            let c = book.contacts.entry(key.clone()).or_insert_with(|| Contact::new(&key, now));
            let (added, dropped) = remember_fact(c, &text, by, now)?;
            if added {
                c.updated_at = now.to_string();
            }
            Ok((Remembered { contact: c.clone(), added, dropped }, added || new))
        })
    }

    /// Forget a contact's fact by its place (from 0). With `expect`, only
    /// when that fact says it: the list may have changed since it was read.
    pub fn forget_fact(&self, number: &str, index: usize, expect: Option<&str>) -> Result<(Contact, Fact), Error> {
        let key = key_of(number)?;
        self.change(|book, now| {
            let c = book.contacts.get_mut(&key).ok_or_else(|| Error::no_contact(&key))?;
            let Some(fact) = c.facts.get(index) else {
                let n = c.facts.len();
                return Err(Error::new(404, "no_fact", format!("there is no fact {index}: this contact has {n} (from 0)")));
            };
            if let Some(want) = expect {
                if fact.text != want.split_whitespace().collect::<Vec<_>>().join(" ") {
                    return Err(Error::new(409, "fact_changed", "the facts changed since they were read: read them again"));
                }
            }
            let fact = c.facts.remove(index);
            c.updated_at = now.to_string();
            Ok(((c.clone(), fact), true))
        })
    }

    /// Every contact as a CSV file, and how many.
    pub fn export_csv(&self) -> Result<(String, usize), Error> {
        let all = self.all()?;
        Ok((csv::export(&all), all.len()))
    }

    /// Read a CSV file into the contacts; with `preview`, only say what it would do.
    pub fn import(&self, request: &csv::ImportRequest) -> Result<csv::ImportReport, Error> {
        let code = request.country.as_deref().map(str::trim).filter(|c| !c.is_empty()).unwrap_or(DEFAULT_COUNTRY);
        let country = country(code).ok_or_else(|| {
            let known: Vec<&str> = COUNTRIES.iter().map(|c| c.code).collect();
            Error::new(400, "bad_country", format!("{code:?} is not a country this knows: one of {}", known.join(", ")))
        })?;
        let table = csv::read_table(&request.csv)?;
        self.change(|book, now| {
            let mut report = csv::merge(book, &table, country, request.replace_names, now);
            report.preview = request.preview;
            let changed = !request.preview && report.added + report.updated > 0;
            Ok((report, changed))
        })
    }
}

static SHARED: OnceLock<Store> = OnceLock::new();

/// Open the contacts in `<data>/callers.json`, upgrading the older file now rather than on the first call.
pub fn init(data_dir: &Path) {
    let store = Store::at(data_dir.join(FILE_NAME));
    if let Err(e) = store.read(|_| Ok(())) {
        log::warn!("contacts: {e}");
    }
    let _ = SHARED.set(store);
}

/// The desktop's contacts (none open before [`init`]: every answer says so).
pub fn shared() -> Store {
    SHARED.get().cloned().unwrap_or_default()
}

/// The name kept for a caller's number.
pub fn name_of(number: &str) -> Option<String> {
    SHARED.get()?.name_of(number)
}

/// A number seen on a call: the caller is a contact from now on (not a hidden one).
pub fn saw(number: &str) {
    if let Some(store) = SHARED.get() {
        if let Err(e) = store.saw(number) {
            log::warn!("contacts: the caller could not be kept: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct Dir(pub PathBuf);

    impl Dir {
        pub fn new(tag: &str) -> Dir {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!("oaiy-contacts-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Dir(dir)
        }
        pub fn store(&self) -> Store {
            Store::at(self.0.join(FILE_NAME))
        }
        pub fn file(&self) -> PathBuf {
            self.0.join(FILE_NAME)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn saved(d: &Dir) -> Value {
        serde_json::from_str(&std::fs::read_to_string(d.file()).unwrap()).unwrap()
    }

    #[test]
    fn the_older_names_file_is_upgraded_in_place_and_kept_once_as_bak() {
        let d = Dir::new("upgrade");
        let old = "{\n  \"491570006\": \"Liam\",\n  \"400000001\": \"  Sam   Lee \"\n}";
        std::fs::write(d.file(), old).unwrap();
        let s = d.store();
        let (all, total) = s.list("").unwrap();
        assert_eq!(total, 2);
        assert_eq!(all.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["Liam", "Sam Lee"]);
        let liam = s.get("0491 570 006").unwrap();
        assert_eq!((liam.key.as_str(), liam.number.as_str(), liam.name_by), ("491570006", "", Some(By::Agent)), "the receptionist wrote the older names");
        assert!(liam.notes.is_empty() && liam.facts.is_empty());
        // Upgraded in place, with the older bytes beside it.
        let v = saved(&d);
        assert_eq!(v["version"], 2);
        assert_eq!(v["contacts"]["491570006"]["name"], "Liam");
        assert_eq!(v["contacts"]["491570006"]["nameBy"], "agent");
        let bak = d.0.join("callers.json.bak");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), old);
        // No temporary file is left behind.
        let leftovers: Vec<String> = std::fs::read_dir(&d.0).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).filter(|n| n.contains(".tmp")).collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");

        // Kept once: an older file seen again later never writes over the first backup.
        std::fs::write(d.file(), "{\"400000002\": \"Kim\"}").unwrap();
        assert_eq!(s.get("0400000002").unwrap().name, "Kim");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), old, "the first backup stays");
    }

    #[test]
    fn a_file_that_cannot_be_read_is_never_written_over() {
        let d = Dir::new("damaged");
        std::fs::write(d.file(), "{ not json").unwrap();
        let s = d.store();
        let e = s.list("").unwrap_err();
        assert_eq!((e.status, e.code), (500, "contacts_unreadable"));
        assert!(s.set("0491570006", Change { name: Some("Liam".into()), ..Change::default() }).is_err());
        assert!(s.saw("0491570006").is_err());
        assert_eq!(s.name_of("0491570006"), None, "the greeting goes on without it");
        assert_eq!(std::fs::read_to_string(d.file()).unwrap(), "{ not json");
        // A list is not a file of contacts either.
        std::fs::write(d.file(), "[1, 2]").unwrap();
        assert_eq!(s.list("").unwrap_err().code, "contacts_unreadable");
    }

    #[test]
    fn a_number_written_any_way_is_one_key() {
        for n in ["+61 491 570 006", "0491 570 006", "+61491570006", "491570006", "(04) 9157-0006", "0061491570006"] {
            assert_eq!(key(n).as_deref(), Some("491570006"), "{n}");
        }
        assert_eq!(key("9876 5432").as_deref(), Some("98765432"), "eight digits is enough");
        assert_eq!(check_number("1234567"), Err(NotANumber::TooShort));
        assert_eq!(check_number("  "), Err(NotANumber::Missing));
        assert_eq!(check_number("N/A"), Err(NotANumber::Missing));
        let d = Dir::new("keys");
        let s = d.store();
        s.set("+61 491 570 006", Change { name: Some("Liam".into()), ..Change::default() }).unwrap();
        for n in ["0491570006", "491570006", "+61491570006", "0061 491 570 006"] {
            assert_eq!(s.get(n).unwrap().name, "Liam", "{n}");
        }
        assert_eq!(s.get("0491570157").unwrap_err().code, "no_contact");
        assert_eq!(s.get("12345").unwrap_err().code, "not_a_phone_number");
    }

    #[test]
    fn hidden_and_withheld_callers_never_become_contacts() {
        let d = Dir::new("hidden");
        let s = d.store();
        for n in ["Private", "Unknown", "anonymous", "Withheld", "No Caller ID", "Restricted", "-1", "-2", "0000000000", "+00000000000", ""] {
            assert_eq!(s.saw(n).unwrap(), None, "{n:?}");
            assert!(s.remember_name(n, "Sam").is_err(), "{n:?}");
            assert!(s.set(n, Change::default()).is_err(), "{n:?}");
            assert!(s.add_fact(n, "likes tea", By::Agent).is_err(), "{n:?}");
        }
        assert_eq!(check_number("Private number"), Err(NotANumber::Hidden));
        assert_eq!(s.set("Private", Change::default()).unwrap_err().code, "hidden");
        assert!(!d.file().exists(), "nothing was written");
    }

    #[test]
    fn a_call_makes_its_caller_a_contact_with_the_number_last_seen() {
        let d = Dir::new("saw");
        let s = d.store();
        let c = s.saw("+61491570006").unwrap().unwrap();
        assert_eq!((c.key.as_str(), c.number.as_str(), c.name.as_str(), c.name_by), ("491570006", "+61491570006", "", None));
        let before = std::fs::metadata(d.file()).unwrap().modified().unwrap();
        // Seen again the same: nothing to save.
        std::thread::sleep(std::time::Duration::from_millis(20));
        s.saw("+61491570006").unwrap();
        assert_eq!(std::fs::metadata(d.file()).unwrap().modified().unwrap(), before);
        // Seen another way: that is the number last seen.
        assert_eq!(s.saw("0491 570 006").unwrap().unwrap().number, "0491 570 006");
        assert_eq!(s.list("").unwrap().1, 1);
    }

    #[test]
    fn single_names_are_names_and_are_never_corrected() {
        let d = Dir::new("single");
        let s = d.store();
        assert_eq!(s.remember_name("+61491570006", "Liam").unwrap(), "Liam");
        assert_eq!(s.name_of("0491570006").as_deref(), Some("Liam"));
        s.set("0400000001", Change { name: Some("  liam ".into()), ..Change::default() }).unwrap();
        assert_eq!(s.get("0400000001").unwrap().name, "liam", "not capitalised");
        s.set("0400000002", Change { name: Some("Zoë  O'Brien-Smith".into()), ..Change::default() }).unwrap();
        assert_eq!(s.get("0400000002").unwrap().name, "Zoë O'Brien-Smith");
        // A name past the limit is cut, not refused (the older route did the same).
        let long = "A".repeat(200);
        assert_eq!(s.remember_name("0400000003", &long).unwrap().chars().count(), MAX_NAME);
        assert!(crate::voice::callers::looks_like_name("Liam"));
    }

    #[test]
    fn a_name_the_person_set_is_never_overwritten_by_the_receptionist() {
        let d = Dir::new("owner");
        let s = d.store();
        // The receptionist learned "Liam Smith" on a call...
        assert_eq!(s.remember_name("+61491570006", "Liam Smith").unwrap(), "Liam Smith");
        assert_eq!(s.get("0491570006").unwrap().name_by, Some(By::Agent));
        // ...and the person says it is "Liam".
        let c = s.set("0491570006", Change { name: Some("Liam".into()), ..Change::default() }).unwrap();
        assert_eq!((c.name.as_str(), c.name_by), ("Liam", Some(By::Owner)));
        // The receptionist hears "Liam Smith" again: the person's name stays, and is the answer.
        assert_eq!(s.remember_name("+61491570006", "Liam Smith").unwrap(), "Liam");
        assert_eq!(s.remember_name("+61491570006", "").unwrap(), "Liam", "nor can it clear it");
        let c = s.get("0491570006").unwrap();
        assert_eq!((c.name.as_str(), c.name_by), ("Liam", Some(By::Owner)));
        assert_eq!(s.name_of("+61491570006").as_deref(), Some("Liam"));
        // The person clearing their name lets the receptionist learn one again.
        let c = s.set("0491570006", Change { name: Some(" ".into()), ..Change::default() }).unwrap();
        assert_eq!((c.name.as_str(), c.name_by), ("", None));
        assert_eq!(s.remember_name("0491570006", "Liam").unwrap(), "Liam");
        assert_eq!(s.get("0491570006").unwrap().name_by, Some(By::Agent));
    }

    #[test]
    fn the_receptionist_clearing_its_name_keeps_the_contact() {
        let d = Dir::new("forget");
        let s = d.store();
        s.remember_name("0491570006", "Liam").unwrap();
        s.set("0491570006", Change { notes: Some("Prefers texts".into()), ..Change::default() }).unwrap();
        s.remember_name("0400000001", "Sam").unwrap();
        assert_eq!(s.remember_name("+61491570006", "  ").unwrap(), "");
        let c = s.get("0491570006").unwrap();
        assert_eq!((c.name.as_str(), c.name_by, c.notes.as_str()), ("", None, "Prefers texts"), "the notes stay");
        assert_eq!(s.name_of("0491570006"), None);
        assert_eq!(s.name_of("0400000001").as_deref(), Some("Sam"), "the others are kept");
        // An empty name for someone unknown makes no contact.
        assert_eq!(s.remember_name("0400000009", "").unwrap(), "");
        assert_eq!(s.get("0400000009").unwrap_err().code, "no_contact");
        // Renamed: the new name replaces the old.
        s.remember_name("0400000001", "Samantha").unwrap();
        assert_eq!(s.name_of("+61400000001").as_deref(), Some("Samantha"));
    }

    #[test]
    fn notes_and_numbers_are_kept_as_the_person_wrote_them_within_their_limits() {
        let d = Dir::new("notes");
        let s = d.store();
        let c = s.set("0491570006", Change { notes: Some("  Prefers mornings.\r\nGate code 1234  ".into()), number: Some("0491 570 006".into()), ..Change::default() }).unwrap();
        assert_eq!(c.notes, "Prefers mornings.\nGate code 1234");
        assert_eq!(c.number, "0491 570 006");
        assert_eq!(c.name_by, None, "no name given, none set");
        let e = s.set("0491570006", Change { notes: Some("x".repeat(MAX_NOTES + 1)), ..Change::default() }).unwrap_err();
        assert_eq!((e.status, e.code), (400, "too_long"));
        assert!(e.message.contains("2000"));
        assert!(s.set("0491570006", Change { notes: Some("x".repeat(MAX_NOTES)), ..Change::default() }).is_ok());
        let e = s.set("0491570006", Change { number: Some("0400 000 001".into()), ..Change::default() }).unwrap_err();
        assert_eq!(e.code, "other_number");
    }

    #[test]
    fn facts_are_added_once_capped_and_forgotten_by_their_place() {
        let d = Dir::new("facts");
        let s = d.store();
        let r = s.add_fact("+61491570006", "  Has a dog   called Max ", By::Agent).unwrap();
        assert!(r.added && r.dropped.is_none());
        assert_eq!(r.contact.facts[0].text, "Has a dog called Max");
        assert_eq!(r.contact.facts[0].by, By::Agent);
        // Said again (in any case): not added twice.
        let r = s.add_fact("0491570006", "has a dog called max", By::Agent).unwrap();
        assert!(!r.added);
        assert_eq!(r.contact.facts.len(), 1);
        // Too long, or empty: refused, saying why.
        let e = s.add_fact("0491570006", &"y".repeat(MAX_FACT + 1), By::Agent).unwrap_err();
        assert_eq!((e.status, e.code), (400, "too_long"));
        assert!(e.message.contains("300"));
        assert_eq!(s.add_fact("0491570006", "   ", By::Agent).unwrap_err().code, "empty_fact");
        assert!(s.add_fact("0491570006", &"y".repeat(MAX_FACT), By::Owner).is_ok());

        // A full list lets its oldest of the receptionist's go.
        for i in 2..MAX_FACTS {
            s.add_fact("0491570006", &format!("fact {i}"), By::Agent).unwrap();
        }
        assert_eq!(s.get("0491570006").unwrap().facts.len(), MAX_FACTS);
        let r = s.add_fact("0491570006", "Moved to Brisbane", By::Agent).unwrap();
        assert!(r.added);
        assert_eq!(r.dropped.unwrap().text, "Has a dog called Max");
        let facts = s.get("0491570006").unwrap().facts;
        assert_eq!(facts.len(), MAX_FACTS);
        assert_eq!(facts.last().unwrap().text, "Moved to Brisbane");
        assert_eq!(facts[0].by, By::Owner, "the person's fact stays");

        // Forgotten by place; the text guards against a list that changed.
        let e = s.forget_fact("0491570006", 1, Some("not this one")).unwrap_err();
        assert_eq!((e.status, e.code), (409, "fact_changed"));
        let (c, gone) = s.forget_fact("0491570006", 1, Some("fact 2")).unwrap();
        assert_eq!(gone.text, "fact 2");
        assert_eq!(c.facts.len(), MAX_FACTS - 1);
        let (_, gone) = s.forget_fact("0491570006", 0, None).unwrap();
        assert_eq!(gone.by, By::Owner);
        assert_eq!(s.forget_fact("0491570006", 99, None).unwrap_err().code, "no_fact");
        assert_eq!(s.forget_fact("0400000001", 0, None).unwrap_err().code, "no_contact");

        // A list of the person's own facts only: full is full.
        let mut c = Contact::new("400000002", "t");
        for i in 0..MAX_FACTS {
            remember_fact(&mut c, &format!("mine {i}"), By::Owner, "t").unwrap();
        }
        assert_eq!(remember_fact(&mut c, "one more", By::Agent, "t").unwrap_err().code, "facts_full");
    }

    #[test]
    fn search_finds_names_notes_facts_and_numbers_typed_any_way() {
        let d = Dir::new("search");
        let s = d.store();
        s.set("0491570006", Change { name: Some("Liam".into()), notes: Some("Owns the café on Smith St".into()), ..Change::default() }).unwrap();
        s.set("0400000001", Change { name: Some("sam".into()), number: Some("+61400000001".into()), ..Change::default() }).unwrap();
        s.set("0298765432", Change { name: Some("Anna".into()), ..Change::default() }).unwrap();
        s.add_fact("0298765432", "Has two border collies", By::Agent).unwrap();
        s.saw("0411222333").unwrap();
        let names = |q: &str| s.list(q).unwrap().0.into_iter().map(|c| if c.name.is_empty() { c.key } else { c.name }).collect::<Vec<_>>();
        // By name, in any case; the nameless after.
        assert_eq!(names(""), ["Anna", "Liam", "sam", "411222333"]);
        assert_eq!(names("LIAM"), ["Liam"]);
        assert_eq!(names("café"), ["Liam"], "the notes");
        assert_eq!(names("collies"), ["Anna"], "what was remembered");
        assert_eq!(names("0491 570"), ["Liam"], "a number the local way, with its 0");
        assert_eq!(names("491570"), ["Liam"]);
        assert_eq!(names("+61 400"), ["sam"]);
        assert_eq!(names("(02) 9876"), ["Anna"]);
        assert_eq!(names("zzz"), Vec::<String>::new());
        assert_eq!(s.list("sam").unwrap().1, 4, "the total is everyone");
    }

    #[test]
    fn local_numbers_are_read_as_the_countrys() {
        let au = country("au").unwrap();
        assert_eq!(international("0491 570 006", au), "+61491570006");
        assert_eq!(international("491570006", au), "+61491570006");
        assert_eq!(international("(02) 9876 5432", au), "+61298765432");
        assert_eq!(international("+61 491 570 006", au), "+61491570006");
        assert_eq!(international("0061 491 570 006", au), "+61491570006");
        assert_eq!(international("61491570006", au), "+61491570006");
        assert_eq!(international("1300 123 456", au), "1300 123 456", "a number it cannot place stays as given");
        let us = country("US").unwrap();
        assert_eq!(international("(415) 555-0100", us), "+14155550100");
        assert_eq!(international("1 415 555 0100", us), "+14155550100");
        assert_eq!(international("011 61 491 570 006", us), "+61491570006");
        assert_eq!(international("07700 900123", country("uk").unwrap()), "+447700900123");
        assert!(country("XX").is_none());
    }
}
