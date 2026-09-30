//! What the dry run says of one thing is made of labelled PARTS, each with a budget of its own.
//!
//! The dry run is where a person decides. A description that is cut as a whole (at 300, 500 or 1,600 characters) can lose what a thing
//! does behind a long address, a padded name or a hundred entries of a collection, and the reviewers found that one place at a time.
//! So there is no whole-item cut. Every kind of thing the dry run describes is a [`Kind`] with:
//!
//! - **fixed parts**, in a fixed order, that always come first: what it does, where it sends, what it may reach, the model, the key,
//!   the permissions. Each is cut on its own, at the budget the kind gives it, and says how long it was when it is cut. A fixed part is
//!   either bounded (a count, a host, a sentence of the code, a value cut inside it, the first few names of a collection and how many
//!   more), and then its budget holds the most it can say and it is never cut, or it holds one free text (an event, a description), and
//!   then it is cut alone: nothing else in it can be pushed out;
//! - **sample parts** after them: free text and collections, a sample and a count, each cut on its own the same way.
//!
//! The kinds, their parts and their budgets are the table [`KINDS`]: nothing that describes a thing can name a part that is not in it
//! (it panics, and every kind is built by a test with every field padded), a part cannot be given a budget anywhere else, BACKUP.md's
//! table of places is made from it, and a test that builds every kind fails for a kind it has no fixture for.
//!
//! Whatever a part says is shown with the characters a person cannot see made visible (see [`visible`]).

use serde::Serialize;

use super::review::RestoreClass;
use super::table::{table, Class};

/// A part of a kind and the most characters of it that are said (a part that is longer is cut, with how long it was).
pub struct PartSpec {
    pub label: &'static str,
    pub budget: usize,
}

const fn part(label: &'static str, budget: usize) -> PartSpec {
    PartSpec { label, budget }
}

/// Fixed parts made from a key table: one for each key of it that acts (class `runs`) under `under`, in the table's order, except the
/// keys that start with one of `except` (its collections are said as samples).
pub struct KeyParts {
    pub table: &'static str,
    pub under: &'static str,
    pub except: &'static [&'static str],
    pub budget: usize,
}

/// A kind of thing the dry run describes.
pub struct Kind {
    pub id: &'static str,
    /// Where it is, for the table in BACKUP.md.
    pub place: &'static str,
    pub fixed: &'static [PartSpec],
    /// Fixed parts that come after `fixed`, made from a key table.
    pub keys: Option<KeyParts>,
    pub sample: &'static [PartSpec],
    /// The most of this kind that one preview describes in full (the rest are named, with how many there are), or none: a kind whose
    /// items are each small.
    pub full: Option<usize>,
}

const NO_KEYS: Option<KeyParts> = None;

/// Every kind of thing the dry run describes.
pub const KINDS: &[Kind] = &[
    Kind { id: "autostart", place: "a service that starts with OAIY (services-autostart.json)", fixed: &[part("starts", 300)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "trigger", place: "a trigger (triggers.json)", fixed: &[part("mode", 80), part("state", 80), part("runs", 240), part("when", 240)], keys: NO_KEYS, sample: &[part("condition", 200)], full: None },
    Kind { id: "ledger", place: "the run history (bridge/ledger.jsonl)", fixed: &[part("records", 300)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "provider-list", place: "a provider of the gateway (ai/providers.json)", fixed: &[part("protocol", 80), part("key", 160), part("local", 120)], keys: NO_KEYS, sample: &[part("address", 190)], full: None },
    Kind { id: "control", place: "the Agent's switch (control.json)", fixed: &[part("switch", 160)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "setup", place: "the setup record (setup.json)", fixed: &[part("accepted", 160)], keys: NO_KEYS, sample: &[part("names", 500)], full: None },
    Kind { id: "agent-model", place: "the Agent's model (agent.json)", fixed: &[part("model", 240)], keys: NO_KEYS, sample: &[], full: None },
    Kind {
        id: "template",
        place: "a service template (templates/)",
        fixed: &[part("autostart", 80), part("replaces", 80), part("runs", 420), part("install", 240), part("writes", 640), part("deletes", 360), part("env", 480), part("cwd", 180), part("marker", 200), part("health", 200), part("docs", 180)],
        keys: NO_KEYS,
        sample: &[],
        full: Some(400),
    },
    Kind { id: "flow", place: "a flow (flows/)", fixed: &[part("steps", 80), part("tool", 200), part("tool-description", 400), part("tool-inputs", 720), part("hook", 240)], keys: NO_KEYS, sample: &[part("kinds", 620)], full: None },
    Kind {
        id: "connector",
        place: "a connector descriptor (connectors/)",
        fixed: &[
            part("replaces", 100),
            part("prefilled", 260),
            part("scopes", 1840),
            part("auth", 1100),
            part("health", 760),
            part("heartbeat", 760),
            part("relay", 1100),
            part("desktopFlows", 1100),
            part("desktopAi", 2000),
            part("flows", 2000),
            part("appLogic", 1300),
            part("dataNode", 940),
            part("scriptProfile", 760),
            part("docs", 760),
        ],
        keys: NO_KEYS,
        sample: &[part("summary", 240), part("other-places", 7400)],
        full: Some(100),
    },
    Kind { id: "callers", place: "what is remembered about callers (callers.json)", fixed: &[part("entries", 320)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "setting", place: "a key that acts in a settings file (the calendar's, a plugin's, the Agent's)", fixed: &[part("sets", 520)], keys: NO_KEYS, sample: &[part("why", 320)], full: None },
    Kind {
        id: "calendar-service",
        place: "a service of the calendar",
        fixed: &[part("about", 200)],
        keys: Some(KeyParts { table: "calendar", under: "settings.services[].", except: &[], budget: 560 }),
        sample: &[],
        full: None,
    },
    Kind {
        id: "calendar-appointment",
        place: "an appointment of the calendar",
        fixed: &[part("about", 300)],
        keys: Some(KeyParts { table: "calendar", under: "appointments[].", except: &[], budget: 560 }),
        sample: &[],
        full: None,
    },
    Kind { id: "voice", place: "a voice clip (voices/)", fixed: &[part("file", 200)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "unreadable", place: "a file that could not be read", fixed: &[part("problem", 480)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "nothing", place: "a settings file that holds nothing OAIY restores", fixed: &[part("nothing", 240)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "more", place: "the things of a kind that are only counted", fixed: &[part("count", 400)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "project", place: "a project of the Agent", fixed: &[part("files", 320)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "brief", place: "the front desk's brief", fixed: &[part("reads", 260), part("says", 800), part("left-out", 300)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "knowledge", place: "a knowledge file of the front desk", fixed: &[part("size", 240), part("left-out", 300)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "desk-files", place: "the other files of the front desk", fixed: &[part("files", 240)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "desk-sessions", place: "the phone's conversations", fixed: &[part("files", 240)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "desk-chat", place: "the front desk's own conversation", fixed: &[part("size", 160)], keys: NO_KEYS, sample: &[], full: None },
    Kind { id: "desk-callers", place: "what the phone's agents remember about people", fixed: &[part("entries", 320), part("left-out", 520)], keys: NO_KEYS, sample: &[], full: None },
    Kind {
        id: "campaign",
        place: "an outreach campaign",
        fixed: &[part("comes-back", 480), part("left-out", 520)],
        keys: Some(KeyParts { table: "agent.campaign", under: "", except: &["collect[]", "people[]", "skipped[]", "id", "slug", "createdAt", "resultsPath", "name"], budget: 640 }),
        sample: &[part("questions", 240), part("question", 900), part("people", 240), part("person", 2000), part("set-aside", 240), part("skipped", 500)],
        full: Some(50),
    },
    Kind { id: "agent-provider", place: "a provider of the Agent", fixed: &[part("type", 80), part("model", 260), part("key", 200), part("beside", 200)], keys: NO_KEYS, sample: &[part("address", 190)], full: None },
];

/// The kind with this id. (An id that is not one is a mistake of whoever wrote the describer: every kind is built by a test.)
pub fn kind(id: &str) -> &'static Kind {
    KINDS.iter().find(|k| k.id == id).unwrap_or_else(|| panic!("the dry run has no kind of thing called {id}"))
}

/// A part of an item, as it was said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartRecord {
    pub label: String,
    pub fixed: bool,
    /// How long it was, when it was cut.
    pub cut_from: Option<usize>,
}

/// One thing a restore could bring back that can act, described for the person. It is built only from parts ([`Parts::item`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReviewItem {
    pub class: RestoreClass,
    /// Where it is in the backup.
    pub name: String,
    pub title: String,
    /// What it does: its parts, one after another.
    pub what: String,
    /// Which kind of thing it is, and how it was said (for the tests: not sent to the dashboard).
    #[serde(skip)]
    pub kind: &'static str,
    #[serde(skip)]
    pub parts: Vec<PartRecord>,
    #[serde(skip)]
    sealed: (),
}

/// A value cut to `budget` characters inside a part (with how long it was, and the characters a person cannot see made visible).
pub fn short(text: &str, budget: usize) -> String {
    cut(text, budget).0
}

/// The text cut to `budget` characters, and how long it was in all when it is cut. The characters a person cannot see are made visible
/// first, so what is counted and shown is what a person reads.
fn bare(text: &str, budget: usize) -> (String, Option<usize>) {
    let seen = visible(text);
    let length = seen.chars().count();
    if length <= budget {
        (seen, None)
    } else {
        (seen.chars().take(budget).collect(), Some(length))
    }
}

/// Text cut to `budget` characters, with how long it was in all when it is cut (the one way the dry run cuts a text).
pub fn cut(text: &str, budget: usize) -> (String, Option<usize>) {
    match bare(text, budget) {
        (seen, None) => (seen, None),
        (seen, Some(length)) => (format!("{seen} … (cut, {length} characters in all)"), Some(length)),
    }
}

/// A value as a sentence says it: in quotes, cut to `budget` characters, and when it is cut how long it was in all, after the quotes.
pub fn quoted(text: &str, budget: usize) -> String {
    match bare(text, budget) {
        (seen, None) => format!("\"{seen}\""),
        (seen, Some(length)) => format!("\"{seen} …\" (cut, {length} characters in all)"),
    }
}

/// The first `most` of a list of lines, each cut on its own to `each` characters, and a last line that says how many more there were.
pub fn lines_of<S: AsRef<str>>(lines: &[S], most: usize, each: usize) -> Vec<String> {
    let mut out: Vec<String> = lines.iter().take(most).map(|l| short(l.as_ref(), each)).collect();
    if lines.len() > most {
        out.push(format!("and {} more are not listed here.", lines.len() - most));
    }
    out
}

/// The first `most` of a list, each cut on its own to `each` characters, said in a line with how many more there were: `a, b and 5 more`.
pub fn some_of<S: AsRef<str>>(items: &[S], most: usize, each: usize) -> String {
    let named = items.iter().take(most).map(|i| short(i.as_ref(), each)).collect::<Vec<_>>().join(", ");
    if items.len() > most {
        format!("{named} and {} more", items.len() - most)
    } else {
        named
    }
}

/// The parts of one item, in the order they are said.
pub struct Parts {
    kind: &'static Kind,
    said: Vec<(PartRecord, String)>,
    rank: Option<usize>,
    sampling: bool,
}

impl Parts {
    pub fn new(id: &str) -> Self {
        Self { kind: kind(id), said: Vec::new(), rank: None, sampling: false }
    }

    /// A fixed part: it is in the kind, comes after the fixed parts said before it, and before every sample. Nothing is said for `None`
    /// (a thing that has no such part).
    pub fn fixed(mut self, label: &str, text: impl AsRef<str>) -> Self {
        assert!(!self.sampling, "{}: the fixed part {label} comes before every sample", self.kind.id);
        let (rank, budget) = match self.kind.fixed.iter().position(|p| p.label == label) {
            Some(at) => (at, self.kind.fixed[at].budget),
            None => match &self.kind.keys {
                Some(keys) if key_position(keys, label).is_some() => (self.kind.fixed.len() + key_position(keys, label).unwrap_or(0), keys.budget),
                _ => panic!("{}: {label} is not a fixed part of it", self.kind.id),
            },
        };
        assert!(self.rank.is_none_or(|last| rank > last), "{}: the fixed part {label} is out of its place or said twice", self.kind.id);
        self.rank = Some(rank);
        let (text, cut_from) = cut(text.as_ref(), budget);
        self.said.push((PartRecord { label: label.to_string(), fixed: true, cut_from }, text));
        self
    }

    /// A sample part (free text, or a collection's sample and count), after the fixed ones. A label may come more than once (the people
    /// of a campaign).
    pub fn sample(mut self, label: &str, text: impl AsRef<str>) -> Self {
        let budget = self.kind.sample.iter().find(|p| p.label == label).unwrap_or_else(|| panic!("{}: {label} is not a sample part of it", self.kind.id)).budget;
        self.sampling = true;
        let (text, cut_from) = cut(text.as_ref(), budget);
        self.said.push((PartRecord { label: label.to_string(), fixed: false, cut_from }, text));
        self
    }

    /// The thing, with the title it is listed under (cut to a hundred and twenty characters, with how long it was).
    pub fn item(self, class: RestoreClass, name: &str, title: &str) -> ReviewItem {
        let what = self.said.iter().filter(|(_, text)| !text.is_empty()).map(|(_, text)| text.as_str()).collect::<Vec<_>>().join(" ");
        ReviewItem {
            class,
            name: name.to_string(),
            title: cut(title, 120).0,
            what,
            kind: self.kind.id,
            parts: self.said.into_iter().map(|(record, _)| record).collect(),
            sealed: (),
        }
    }
}

/// The position of a key of the table among the fixed parts made from it, or none when it is not one.
fn key_position(keys: &KeyParts, path: &str) -> Option<usize> {
    key_paths(keys).iter().position(|p| p == path)
}

/// The table of BACKUP.md that says, for every kind of thing the dry run describes, its fixed parts (in the order they are said, each with
/// the most it says of it), its sample parts, and how many of it are described in full. It is made from [`KINDS`] (and from the key tables of
/// the kinds that say the keys of one), so it cannot be out of date.
pub fn render_places() -> String {
    let mut out = String::from("| Place | Fixed parts, said first and in this order (budget) | Sample parts, said after them (budget) | Described in full |\n|---|---|---|---|\n");
    let said = |specs: &[PartSpec]| specs.iter().map(|p| format!("`{}` ({})", p.label, p.budget)).collect::<Vec<_>>().join(", ");
    for kind in KINDS {
        let mut fixed = said(kind.fixed);
        if let Some(keys) = &kind.keys {
            let paths = key_paths(keys);
            let more = format!("then each of the {} keys of the table `{}` that acts{} ({} each)", paths.len(), keys.table, if keys.under.is_empty() { String::new() } else { format!(" under `{}`", keys.under) }, keys.budget);
            fixed = if fixed.is_empty() { more } else { format!("{fixed}, {more}") };
        }
        let sample = said(kind.sample);
        let full = kind.full.map(|n| format!("{n}, the rest named and counted")).unwrap_or_else(|| "all (each is small)".to_string());
        out.push_str(&format!("| {} (`{}`) | {} | {} | {} |\n", kind.place, kind.id, if fixed.is_empty() { "-".to_string() } else { fixed }, if sample.is_empty() { "-".to_string() } else { sample }, full));
    }
    out
}

/// The keys of a table that a kind says as fixed parts, in the table's order: the keys that act under `under`, less the containers and the
/// keys `except` names (a name is that key; a name that ends in `]` is every key of that collection).
pub fn key_paths(keys: &KeyParts) -> Vec<String> {
    let Some(kt) = table().key_table(keys.table) else { return Vec::new() };
    let is_container = |k: &super::table::KeyRow| matches!(k.ty, Some(super::table::ValueType::Object | super::table::ValueType::Objects { .. }));
    kt.keys.iter().filter(|k| k.class == Class::Runs && !is_container(k) && k.path.starts_with(keys.under) && !keys.except.iter().any(|e| k.path == *e || (e.ends_with(']') && k.path.starts_with(e)))).map(|k| k.path.clone()).collect()
}

/// Whether a character is one a person cannot see (or that is drawn as nothing): a control, a format character (a zero-width or
/// direction control, a joiner, a variation selector, a tag), a filler, or a line or paragraph separator.
pub fn is_invisible(c: char) -> bool {
    matches!(c as u32,
        0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1F | 0x7F..=0x9F
        | 0xAD | 0x034F | 0x061C | 0x115F | 0x1160 | 0x17B4 | 0x17B5 | 0x180B..=0x180F
        | 0x200B..=0x200F | 0x2028..=0x202E | 0x2060..=0x206F | 0x3164 | 0xFE00..=0xFE0F | 0xFEFF | 0xFFA0 | 0xFFF0..=0xFFF8
        | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A | 0xE0000..=0xE0FFF)
}

/// Text with the characters a person cannot see made visible: each run of them is said as `[3 invisible characters: U+E0041 U+E0042
/// U+E0043]` (the first six named), so a person reads that they are there and how many, and what they are.
pub fn visible(text: &str) -> String {
    if !text.chars().any(is_invisible) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut run: Vec<char> = Vec::new();
    let flush = |run: &mut Vec<char>, out: &mut String| {
        if run.is_empty() {
            return;
        }
        let named = run.iter().take(6).map(|c| format!("U+{:04X}", *c as u32)).collect::<Vec<_>>().join(" ");
        out.push_str(&format!("[{} invisible character{}: {named}{}]", run.len(), if run.len() == 1 { "" } else { "s" }, if run.len() > 6 { " …" } else { "" }));
        run.clear();
    };
    for c in text.chars() {
        if is_invisible(c) {
            run.push(c);
        } else {
            flush(&mut run, &mut out);
            out.push(c);
        }
    }
    flush(&mut run, &mut out);
    out
}

/// The most notes of one class that a restore says (the rest of the class are counted in a note of their own).
pub const MOST_NOTES_PER_CLASS: usize = 8;

/// The notes a restore says of what it left out, cleaned or changed, kept by class. A class has a budget of its own: a backup that makes
/// hundreds of notes of one class (a campaign for each, a file for each) says the first few and how many more, and cannot crowd out the
/// notes of another class, whatever room the whole list has.
#[derive(Default)]
pub struct NoteBook {
    classes: Vec<(&'static str, Vec<String>, usize)>,
}

impl NoteBook {
    pub fn push(&mut self, class: &'static str, note: impl Into<String>) {
        let at = match self.classes.iter().position(|(c, _, _)| *c == class) {
            Some(at) => at,
            None => {
                self.classes.push((class, Vec::new(), 0));
                self.classes.len() - 1
            }
        };
        if self.classes[at].1.len() < MOST_NOTES_PER_CLASS {
            self.classes[at].1.push(note.into());
        } else {
            self.classes[at].2 += 1;
        }
    }

    pub fn extend(&mut self, class: &'static str, notes: impl IntoIterator<Item = String>) {
        for note in notes {
            self.push(class, note);
        }
    }

    /// The notes, class by class, each class's followed by how many more of it there were.
    pub fn finish(self) -> Vec<String> {
        let mut out = Vec::new();
        for (class, notes, over) in self.classes {
            out.extend(notes);
            if over > 0 {
                out.push(format!("{over} more note{} of this kind ({class}) {} not listed here.", if over == 1 { "" } else { "s" }, if over == 1 { "is" } else { "are" }));
            }
        }
        out
    }

    /// How many notes there are, at the most, when there are `classes` classes.
    pub const fn most(classes: usize) -> usize {
        classes * (MOST_NOTES_PER_CLASS + 1)
    }
}

/// The preview of a kind that is described in full only up to a number ([`Kind::full`]): the items beyond it are named, with how many
/// there are, and not described. (A hostile backup of three hundred campaigns as long as a campaign can be would otherwise make a
/// preview of many megabytes.)
pub fn cap_full(items: Vec<ReviewItem>) -> Vec<ReviewItem> {
    let mut total: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    for item in &items {
        *total.entry(item.kind).or_default() += 1;
    }
    let mut seen: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    items
        .into_iter()
        .map(|item| {
            let n = seen.entry(item.kind).or_default();
            *n += 1;
            match kind(item.kind).full {
                Some(limit) if *n > limit => Parts::new("more")
                    .fixed("count", format!("One of {} of its kind in this backup ({}): only the first {limit} are described in full, so this one is named and not described. Read it (its own file, or in the Agent) before you use it.", total[item.kind], kind(item.kind).place))
                    .item(item.class, &item.name, &item.title),
                _ => item,
            }
        })
        .collect()
}

/// Why a text is not one that comes back (a value a person could have typed to be read by a model or a caller), or `None`. The dry run
/// makes what a person cannot see visible ([`visible`]); this is the other half, for a value that has been made to say one thing to a person
/// and another to a model. It does not strip anything: a value that holds hidden text is not brought back, and is said not to be. What is
/// refused, decided and tested both ways (flags, joined emoji, right-to-left and Persian text pass):
///
/// - a control character (but a line break or a tab);
/// - a text-direction override, embedding or isolate (U+202A-202E, U+2066-2069): they make what is read differ from what is shown;
/// - a tag character (U+E0000-E007F) that is not the end of a flag: a flag of a region is U+1F3F4, two to seven tag letters or digits and the
///   cancel tag, and nothing else made of tags is text a person can read;
/// - more than three characters that are invisible and that nothing explains (zero-width spaces, word joiners, invisible operators, fillers,
///   variation selectors beyond the emoji ones): a byte order mark or a stray one is not refused, a message hidden in them is;
/// - a run of six or more joiners, selectors and direction marks in a row (an emoji sequence puts an emoji between two of them);
/// - a text that is mostly invisible characters (more than eight, and more than half).
///
/// A zero-width joiner, an emoji variation selector and a left-to-right or right-to-left mark are explained (emoji sequences and right-to-left
/// text need them); so is a zero-width non-joiner between two letters that are not ASCII (Persian and other scripts need it).
pub fn text_problem(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.iter().any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t')) {
        return Some("it has control characters".to_string());
    }
    if chars.iter().any(|c| matches!(*c as u32, 0x202A..=0x202E | 0x2066..=0x2069)) {
        return Some("it holds a text-direction override, embedding or isolate, which makes what is read differ from what is shown".to_string());
    }
    // Tag characters: only the end of a flag.
    let is_tag = |c: char| matches!(c as u32, 0xE0000..=0xE007F);
    let mut i = 0;
    while i < chars.len() {
        if !is_tag(chars[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && is_tag(chars[i]) {
            i += 1;
        }
        let run = &chars[start..i];
        let flag = start > 0 && chars[start - 1] == '\u{1F3F4}' && (3..=8).contains(&run.len()) && run.last() == Some(&'\u{E007F}') && run[..run.len() - 1].iter().all(|c| matches!(*c as u32, 0xE0030..=0xE0039 | 0xE0061..=0xE007A));
        if !flag {
            return Some(format!("it holds {} tag characters (invisible text) that are not the end of a flag", run.len()));
        }
    }
    let explained = |at: usize| -> bool {
        match chars[at] as u32 {
            0x200D | 0x200E | 0x200F | 0xFE0E | 0xFE0F => true,
            0x200C => at > 0 && at + 1 < chars.len() && chars[at - 1].is_alphabetic() && !chars[at - 1].is_ascii() && chars[at + 1].is_alphabetic() && !chars[at + 1].is_ascii(),
            // A tag of a flag was passed above.
            0xE0000..=0xE007F => true,
            _ => false,
        }
    };
    let invisible = |c: char| is_invisible(c) && !matches!(c, '\n' | '\r' | '\t');
    let hidden = (0..chars.len()).filter(|&at| invisible(chars[at]) && !explained(at)).count();
    if hidden > 3 {
        return Some(format!("it holds {hidden} characters a person cannot see (zero-width or format characters)"));
    }
    let joiner = |c: char| matches!(c as u32, 0x200C..=0x200F | 0xFE0E | 0xFE0F | 0xAD);
    let (mut run, mut longest) = (0usize, 0usize);
    for &c in &chars {
        run = if joiner(c) { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    if longest >= 6 {
        return Some(format!("it holds a run of {longest} joiners, selectors or direction marks that is not an emoji sequence"));
    }
    let total = chars.iter().filter(|c| invisible(**c)).count();
    if total > 8 && total * 2 > chars.len() {
        return Some("most of it is characters a person cannot see".to_string());
    }
    None
}