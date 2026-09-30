//! The owner's settings for transferring calls and taking messages: `<data>/ring.json`.
//!
//! Everything here is off until the owner turns it on, so a desktop that has never
//! seen this file answers calls exactly as before: `enabled` ("Transfer calls to
//! me") and `takeMessages` are both false, and no tool is offered to the model.
//! Taking messages is the fallback of every transfer that is not answered, so it
//! cannot be off while transfers are on ([`RingSettings::sanitize`] turns it on).
//!
//! The file holds numbers (the owner's VIP list), so it is written through
//! [`crate::secret_file`]: owner-only on Linux and macOS, and with the data folder's own permissions on Windows, where nothing narrows it. It is
//! read as UTF-8 or as UTF-16 with its byte order mark. A file that cannot be used (not text, not settings, or from a newer OAIY) is put aside as
//! `ring.json.corrupt` (`.corrupt.1`, and so on) and the desktop starts with everything off: a broken file never turns a transfer on. A file that
//! cannot be read at all is left as it is, everything is off, and no change is written over it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// The settings file's name in the data folder.
pub const FILE_NAME: &str = "ring.json";
/// The file's shape this desktop writes.
pub const VERSION: u64 = 1;

/// When the receptionist may put a caller through on its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Initiative {
    /// Only when the caller asks for a person.
    #[default]
    OnRequest,
    /// Also when the caller says one of the owner's urgent phrases.
    OnRequestOrUrgent,
}

/// When phones ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneRing {
    #[default]
    WhenAway,
    Always,
    Never,
}

/// When this computer rings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopRing {
    /// While the owner is at it.
    #[default]
    Auto,
    Always,
    Never,
}

/// Whether the owner is away.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Away {
    /// By whether they are at the computer.
    #[default]
    Auto,
    On,
    Off,
}

/// Hours nobody is rung, by the owner's clock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct QuietHours {
    pub enabled: bool,
    /// `HH:MM`.
    pub start: String,
    /// `HH:MM`; a window that ends before it starts crosses midnight.
    pub end: String,
    /// The days it starts on, one bit each (bit 0 Sunday to bit 6 Saturday).
    pub days: u8,
    /// An urgent request the owner allowed still rings.
    pub allow_urgent: bool,
    /// A VIP still rings.
    pub allow_vip: bool,
}

impl Default for QuietHours {
    fn default() -> Self {
        Self { enabled: false, start: "21:00".into(), end: "07:00".into(), days: 127, allow_urgent: false, allow_vip: true }
    }
}

/// How often a caller may be put through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Limits {
    /// Tries on one call.
    pub per_call: u32,
    /// Seconds between two tries on one call.
    pub gap_seconds: u64,
    /// Tries for one caller in an hour (a VIP is exempt).
    pub per_caller_hour: u32,
    /// Tries in an hour, all callers.
    pub global_hour: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self { per_call: 2, gap_seconds: 60, per_caller_hour: 3, global_hour: 10 }
    }
}

/// The owner's settings. See [`FILE_NAME`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
pub struct RingSettings {
    /// "Transfer calls to me": the receptionist may try to reach the owner for a caller who asks.
    pub enabled: bool,
    /// "Take messages": the receptionist may record a message for the owner.
    pub take_messages: bool,
    pub initiative: Initiative,
    /// Phrases (2 to 40 characters, at most 20) that make a request urgent, when `initiative` allows it.
    pub urgent_phrases: Vec<String>,
    /// How long a ring lasts, seconds (20 to 90).
    pub ring_seconds: u32,
    pub phone_ring: PhoneRing,
    pub desktop_ring: DesktopRing,
    pub away: Away,
    /// Unix seconds: a timed `away: on` ends then.
    pub away_until: Option<u64>,
    /// How recent the owner's last input is for them to count as at the computer, seconds (30 to 900).
    pub desktop_active_seconds: u32,
    pub quiet_hours: QuietHours,
    /// Numbers that ring whatever the limits and quiet hours say (at most 50), compared by their last nine digits.
    pub vip_numbers: Vec<String>,
    pub limits: Limits,
    /// Approved companions that are the Windows Companion on this computer (by thumbprint).
    pub windows_companions: Vec<String>,
    /// Approved companions never rung (by thumbprint).
    pub excluded_devices: Vec<String>,
}

impl Default for RingSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            take_messages: false,
            initiative: Initiative::OnRequest,
            urgent_phrases: Vec::new(),
            ring_seconds: super::plan::RING_DEFAULT,
            phone_ring: PhoneRing::WhenAway,
            desktop_ring: DesktopRing::Auto,
            away: Away::Auto,
            away_until: None,
            desktop_active_seconds: 120,
            quiet_hours: QuietHours::default(),
            vip_numbers: Vec::new(),
            limits: Limits::default(),
            windows_companions: Vec::new(),
            excluded_devices: Vec::new(),
        }
    }
}

const MAX_URGENT_PHRASES: usize = 20;
const MAX_VIP_NUMBERS: usize = 50;
const MAX_DEVICES: usize = 50;

impl RingSettings {
    /// Bring every value inside what it may be: ring time, active window, limits and lists are
    /// clamped, quiet hours that are not `HH:MM` fall back to their defaults, and taking messages
    /// is on whenever transfers are.
    pub fn sanitize(&mut self) {
        use super::plan::{RING_MAX, RING_MIN};
        self.ring_seconds = self.ring_seconds.clamp(RING_MIN, RING_MAX);
        self.desktop_active_seconds = self.desktop_active_seconds.clamp(30, 900);
        let defaults = QuietHours::default();
        if !is_hhmm(&self.quiet_hours.start) {
            self.quiet_hours.start = defaults.start.clone();
        }
        if !is_hhmm(&self.quiet_hours.end) {
            self.quiet_hours.end = defaults.end.clone();
        }
        self.quiet_hours.start = self.quiet_hours.start.trim().to_string();
        self.quiet_hours.end = self.quiet_hours.end.trim().to_string();
        self.quiet_hours.days &= 0b111_1111;
        let l = &mut self.limits;
        l.per_call = l.per_call.clamp(1, 5);
        l.gap_seconds = l.gap_seconds.clamp(10, 600);
        l.per_caller_hour = l.per_caller_hour.clamp(1, 10);
        l.global_hour = l.global_hour.clamp(1, 100);
        self.urgent_phrases = self
            .urgent_phrases
            .iter()
            .map(|p| p.trim().to_string())
            .filter(|p| (2..=40).contains(&p.chars().count()))
            .take(MAX_URGENT_PHRASES)
            .collect();
        let mut vips: Vec<String> = Vec::new();
        for number in &self.vip_numbers {
            let number = number.trim().to_string();
            // A number with no key (too short, hidden) could never match a caller: it is not kept.
            let Some(key) = crate::voice::contacts::key(&number) else { continue };
            if !vips.iter().any(|v| crate::voice::contacts::key(v).as_deref() == Some(key.as_str())) {
                vips.push(number);
            }
        }
        vips.truncate(MAX_VIP_NUMBERS);
        self.vip_numbers = vips;
        for list in [&mut self.windows_companions, &mut self.excluded_devices] {
            let mut seen: Vec<String> = Vec::new();
            for id in list.iter() {
                let id = id.trim().to_string();
                if !id.is_empty() && id.len() <= 128 && !seen.contains(&id) {
                    seen.push(id);
                }
            }
            seen.truncate(MAX_DEVICES);
            *list = seen;
        }
        if self.enabled {
            // The fallback of a transfer nobody takes is a message: there is no transfer without it.
            self.take_messages = true;
        }
    }

    /// Whether the receptionist may offer to take a message (also whenever transfers are on).
    pub fn messages_on(&self) -> bool {
        self.take_messages || self.enabled
    }

    /// Whether `number` is one the owner named as a VIP.
    pub fn is_vip(&self, number: &str) -> bool {
        let Some(key) = crate::voice::contacts::key(number) else { return false };
        self.vip_numbers.iter().any(|v| crate::voice::contacts::key(v).as_deref() == Some(key.as_str()))
    }

    /// `away` as it stands at `now_unix`: a timed `on` that has ended is `auto` again.
    pub fn away_at(&self, now_unix: u64) -> Away {
        match (self.away, self.away_until) {
            (Away::On, Some(until)) if now_unix >= until => Away::Auto,
            (away, _) => away,
        }
    }
}

fn is_hhmm(text: &str) -> bool {
    let Some((h, m)) = text.trim().split_once(':') else { return false };
    matches!((h.trim().parse::<u32>(), m.trim().parse::<u32>()), (Ok(h), Ok(m)) if h < 24 && m < 60)
}

/// Why a change to the settings was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsError(pub String);

impl std::fmt::Display for SettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The settings, in memory and on disk. Cloning shares them.
#[derive(Clone)]
pub struct SettingsStore {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    path: Option<PathBuf>,
    settings: RingSettings,
    /// Why the file was not used, in plain words (and where it is kept), when it was not.
    problem: Option<String>,
    /// The file could not be read, or could not be put aside: it is all that is left of what the owner set, so no
    /// change is written over it. A file that could not be read is read again (see `retry`), and is no longer protected when it can be.
    protected: bool,
    patience: crate::secret_file::Patience,
    /// Set while the file could not be read: when it is read again. Everything stays off meanwhile.
    retry: Option<crate::secret_file::Retry>,
}

impl SettingsStore {
    /// Settings kept in `<dir>/ring.json` (UTF-8, or UTF-16 with its byte order mark, as another program may have saved it),
    /// read now. No file: everything off. A file that is not settings, or is not text: put aside as `ring.json.corrupt`
    /// (`.corrupt.1`, and so on: no earlier one is replaced), everything off. A file that cannot be read, or put aside:
    /// everything off, and no change is written over it (a change is refused and says why).
    pub fn open(dir: &Path) -> Self {
        Self::open_patiently(dir, crate::secret_file::Patience::default())
    }

    /// [`SettingsStore::open`], waiting for a file that is busy as `patience` says: it is read again after each of its short pauses, and if it is still
    /// busy everything stays off (never on for want of a read), the owner is told, and it is read again every so often until it can be. A transient lock at
    /// start-up (an antivirus scan, an indexer) does not leave the settings off until the next start.
    pub fn open_patiently(dir: &Path, patience: crate::secret_file::Patience) -> Self {
        let path = dir.join(FILE_NAME);
        let mut inner = Inner { path: Some(path.clone()), settings: RingSettings::default(), problem: None, protected: false, patience: patience.clone(), retry: None };
        inner.load(&path, &patience.start);
        Self { inner: Arc::new(Mutex::new(inner)) }
    }

    /// Settings held in memory only (tests, and a desktop whose data folder is not known).
    pub fn in_memory(settings: RingSettings) -> Self {
        let mut settings = settings;
        settings.sanitize();
        Self { inner: Arc::new(Mutex::new(Inner { path: None, settings, problem: None, protected: false, patience: crate::secret_file::Patience::default(), retry: None })) }
    }

    /// In plain words, why the settings file was not used at the start (and where it is kept), or None.
    pub fn load_problem(&self) -> Option<String> {
        self.lock().problem.clone()
    }

    /// The settings, read again first if their file was busy and it is time to try (each use is a chance to).
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.retry.as_ref().is_some_and(|r| r.due()) {
            inner.recover();
        }
        inner
    }

    pub fn get(&self) -> RingSettings {
        self.lock().settings.clone()
    }

    /// Change some of the settings: `change` is an object of the settings' own names (any of them),
    /// with `quietHours` and `limits` changed member by member. A name that is not a setting, or a
    /// value of the wrong kind, refuses the whole change. Answers the settings now.
    pub fn change(&self, change: &Value) -> Result<RingSettings, SettingsError> {
        let Some(change) = change.as_object() else {
            return Err(SettingsError("the settings are an object".into()));
        };
        let mut inner = self.lock();
        let mut merged = serde_json::to_value(&inner.settings).map_err(|e| SettingsError(e.to_string()))?;
        merge(&mut merged, change);
        let mut next: RingSettings = serde_json::from_value(merged).map_err(|e| SettingsError(format!("that is not a valid setting: {e}")))?;
        next.sanitize();
        if let Some(path) = &inner.path {
            if inner.protected {
                return Err(SettingsError(format!("the settings could not be saved: {FILE_NAME} could not be read or put aside, and is not written over")));
            }
            let mut file =serde_json::to_value(&next).map_err(|e| SettingsError(e.to_string()))?;
            file["version"] = json!(VERSION);
            let body = serde_json::to_string_pretty(&file).map_err(|e| SettingsError(e.to_string()))?;
            crate::secret_file::write(path, body).map_err(|e| SettingsError(format!("the settings could not be saved: {e}")))?;
        }
        inner.settings = next.clone();
        Ok(next)
    }
}

impl Inner {
    /// Read the file into the settings, waiting as `pauses` say for one that is busy. Everything stays off until it is read.
    fn load(&mut self, path: &Path, pauses: &[std::time::Duration]) {
        use crate::secret_file::{read_text_patiently, Retry, Text};
        let why = match read_text_patiently(path, pauses) {
            Text::Missing => None,
            Text::Text(text) => match parse(&text) {
                Ok(settings) => {
                    self.settings = settings;
                    None
                }
                Err(why) => Some(why),
            },
            Text::Undecodable(why) => Some(format!("it is not text ({why})")),
            Text::Unreadable(e) => {
                let first = self.retry.is_none();
                match &mut self.retry {
                    Some(retry) => retry.failed(),
                    None => self.retry = Some(Retry::began(&self.patience)),
                }
                if first {
                    log::warn!("ring settings: {} could not be read ({e}); everything stays off and nothing will be written over it, and it is read again every so often", path.display());
                }
                self.protected = true;
                self.problem = Some(format!("{FILE_NAME} could not be read ({e}): another program may have it open. OAIY is trying again, so everything stays off, and a change is refused, until it can be read. It has not been changed."));
                return;
            }
        };
        if let Some(why) = why {
            match crate::secret_file::keep_aside(path) {
                Ok(aside) => {
                    let name = aside.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    log::warn!("ring settings: {} is not usable ({why}); it is kept as {name} and everything stays off", path.display());
                    self.problem = Some(format!("{FILE_NAME} could not be used ({why}), so everything is off. It is kept as {name} beside it."));
                }
                Err(e) => {
                    log::warn!("ring settings: {} is not usable ({why}) and could not be put aside ({e}); everything stays off and nothing will be written over it", path.display());
                    self.protected = true;
                    self.problem = Some(format!("{FILE_NAME} could not be used ({why}) and could not be put aside ({e}), so everything is off, and a change is refused until that is put right. It has not been changed."));
                }
            }
        }
        if self.retry.take().is_some() {
            log::info!("ring settings: {} could be read again", path.display());
        }
    }

    /// The file could not be read at the last try, and it is time to read it again.
    fn recover(&mut self) {
        let Some(path) = self.path.clone() else { return };
        self.protected = false;
        self.problem = None;
        self.load(&path, &[]);
    }
}

/// The settings a file says, or why it is not.
fn parse(text: &str) -> Result<RingSettings, String> {
    let mut value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let Some(object) = value.as_object_mut() else { return Err("not an object".into()) };
    match object.remove("version") {
        None => {}
        Some(v) if v.as_u64().is_some_and(|v| v <= VERSION) => {}
        Some(v) => return Err(format!("version {v} is from a newer OAIY")),
    }
    let mut settings: RingSettings = serde_json::from_value(value).map_err(|e| e.to_string())?;
    settings.sanitize();
    Ok(settings)
}

/// Lay `change` over `base`: an object member by member for the two nested groups, anything else replaced.
fn merge(base: &mut Value, change: &Map<String, Value>) {
    let Some(base) = base.as_object_mut() else { return };
    for (key, value) in change {
        if matches!(key.as_str(), "quietHours" | "limits") {
            if let (Some(Value::Object(inner)), Value::Object(more)) = (base.get_mut(key), value) {
                for (k, v) in more {
                    inner.insert(k.clone(), v.clone());
                }
                continue;
            }
        }
        base.insert(key.clone(), value.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret_file::testing::{assert_private, TempDir};

    #[test]
    fn a_desktop_that_never_saw_the_file_has_everything_off() {
        let dir = TempDir::new("ring-fresh");
        let s = SettingsStore::open(&dir.0).get();
        assert!(!s.enabled && !s.take_messages && !s.messages_on());
        assert_eq!(s, RingSettings::default());
        assert!(!s.quiet_hours.enabled, "quiet hours stay off: the desktop does not know the business's hours");
        assert_eq!((s.ring_seconds, s.desktop_active_seconds), (40, 120));
        assert_eq!(s.limits, Limits { per_call: 2, gap_seconds: 60, per_caller_hour: 3, global_hour: 10 });
        assert!(!dir.0.join(FILE_NAME).exists(), "opening writes nothing");
    }

    #[test]
    fn settings_round_trip_through_an_owner_only_file() {
        let dir = TempDir::new("ring-roundtrip");
        let store = SettingsStore::open(&dir.0);
        let kept = store.change(&json!({"enabled": true, "ringSeconds": 55, "vipNumbers": ["0491 570 006"], "quietHours": {"enabled": true, "start": "22:00"}})).unwrap();
        assert!(kept.enabled && kept.take_messages, "transfers on turn messages on: the fallback exists");
        assert_eq!(kept.ring_seconds, 55);
        assert!(kept.quiet_hours.enabled && kept.quiet_hours.start == "22:00" && kept.quiet_hours.end == "07:00", "a nested change keeps what it did not name");
        assert_private(&dir.0.join(FILE_NAME));
        let again = SettingsStore::open(&dir.0).get();
        assert_eq!(again, kept);
        let file: Value = serde_json::from_str(&std::fs::read_to_string(dir.0.join(FILE_NAME)).unwrap()).unwrap();
        assert_eq!(file["version"], 1);
    }

    #[test]
    fn values_are_clamped_and_lists_kept_short() {
        let dir = TempDir::new("ring-clamp");
        let store = SettingsStore::open(&dir.0);
        let s = store
            .change(&json!({"ringSeconds": 5, "desktopActiveSeconds": 5000, "limits": {"perCall": 99, "gapSeconds": 1, "perCallerHour": 0, "globalHour": 9999},
                "quietHours": {"start": "25:99", "end": "nonsense", "days": 255},
                "urgentPhrases": ["x", "gas leak", "  burst pipe  ", "a phrase that is far too long to be kept because it goes on and on"],
                "vipNumbers": ["0491 570 006", "+61491570006", "12", "Private"]}))
            .unwrap();
        assert_eq!((s.ring_seconds, s.desktop_active_seconds), (20, 900));
        assert_eq!(s.limits, Limits { per_call: 5, gap_seconds: 10, per_caller_hour: 1, global_hour: 100 });
        assert_eq!((s.quiet_hours.start.as_str(), s.quiet_hours.end.as_str(), s.quiet_hours.days), ("21:00", "07:00", 127));
        assert_eq!(s.urgent_phrases, vec!["gas leak".to_string(), "burst pipe".to_string()]);
        assert_eq!(s.vip_numbers, vec!["0491 570 006".to_string()], "one person once, and numbers that could never match are not kept");
        let many: Vec<String> = (0..80).map(|n| format!("0491 570 {n:03}")).collect();
        assert_eq!(store.change(&json!({"vipNumbers": many})).unwrap().vip_numbers.len(), 50);
        assert_eq!(store.change(&json!({"ringSeconds": 500})).unwrap().ring_seconds, 90);
    }

    #[test]
    fn a_name_that_is_not_a_setting_or_a_value_of_the_wrong_kind_refuses_the_whole_change() {
        let store = SettingsStore::in_memory(RingSettings::default());
        assert!(store.change(&json!({"enabled": true, "typo": 1})).is_err());
        assert!(!store.get().enabled, "nothing of a refused change is kept");
        assert!(store.change(&json!({"phoneRing": "sometimes"})).is_err());
        assert!(store.change(&json!({"ringSeconds": "forty"})).is_err());
        assert!(store.change(&json!([1])).is_err());
        assert_eq!(store.get(), RingSettings::default());
    }

    #[test]
    fn a_file_that_is_not_settings_is_put_aside_and_everything_is_off() {
        for (tag, text) in [("garbage", "{not json"), ("array", "[1,2]"), ("wrong-kind", r#"{"enabled": "yes"}"#), ("newer", r#"{"version": 99, "enabled": true}"#), ("unknown", r#"{"enabled": true, "surprise": 1}"#)] {
            let dir = TempDir::new(&format!("ring-corrupt-{tag}"));
            std::fs::write(dir.0.join(FILE_NAME), text).unwrap();
            let store = SettingsStore::open(&dir.0);
            assert!(!store.get().enabled, "{tag}: a broken file never turns a transfer on");
            assert_eq!(store.get(), RingSettings::default(), "{tag}");
            assert_eq!(std::fs::read_to_string(dir.0.join("ring.json.corrupt")).unwrap(), text, "{tag}: the original is kept");
            assert!(!dir.0.join(FILE_NAME).exists(), "{tag}");
        }
    }

    #[test]
    fn a_settings_file_saved_by_another_program_as_utf16_or_with_a_byte_order_mark_is_read_and_not_lost() {
        let dir = TempDir::new("ring-utf16");
        let text = r#"{"version": 1, "enabled": true, "ringSeconds": 55}"#;
        let le: Vec<u8> = [0xFF, 0xFE].into_iter().chain(text.encode_utf16().flat_map(|u| u.to_le_bytes())).collect();
        let be: Vec<u8> = [0xFE, 0xFF].into_iter().chain(text.encode_utf16().flat_map(|u| u.to_be_bytes())).collect();
        let utf8_with_mark: Vec<u8> = [0xEF, 0xBB, 0xBF].into_iter().chain(text.bytes()).collect();
        for (tag, bytes) in [("utf-16 le", le), ("utf-16 be", be), ("utf-8 with a mark", utf8_with_mark)] {
            std::fs::write(dir.0.join(FILE_NAME), &bytes).unwrap();
            let store = SettingsStore::open(&dir.0);
            assert_eq!((store.get().enabled, store.get().ring_seconds), (true, 55), "{tag}: the owner's own settings are read");
            assert!(store.load_problem().is_none(), "{tag}");
            assert!(!dir.0.join("ring.json.corrupt").exists(), "{tag}: it was read, not put aside");
            // A change keeps the rest, and the file is written again as UTF-8.
            store.change(&json!({"ringSeconds": 60})).unwrap();
            assert_eq!(SettingsStore::open(&dir.0).get().ring_seconds, 60, "{tag}");
        }
    }

    #[test]
    fn a_settings_file_that_cannot_be_decoded_is_kept_as_it_was_and_never_written_over() {
        let dir = TempDir::new("ring-undecodable");
        let first: &[u8] = &[b'{', 0xC3, 0x28, b'"', 0xFF, b'}'];
        std::fs::write(dir.0.join(FILE_NAME), first).unwrap();
        let store = SettingsStore::open(&dir.0);
        assert_eq!(store.get(), RingSettings::default(), "everything stays off");
        assert!(store.load_problem().is_some_and(|p| p.contains("ring.json.corrupt")), "{:?}", store.load_problem());
        assert_eq!(std::fs::read(dir.0.join("ring.json.corrupt")).unwrap(), first, "the original is kept as it was");
        // The owner sets things again: that is a file of its own, and the original is untouched.
        store.change(&json!({"enabled": true})).unwrap();
        assert_eq!(std::fs::read(dir.0.join("ring.json.corrupt")).unwrap(), first);
        assert!(SettingsStore::open(&dir.0).get().enabled);
        // UTF-16 with a lone surrogate: no text can be made of it, and the first is not replaced by it.
        let second: Vec<u8> = vec![0xFF, 0xFE, 0x7B, 0x00, 0x00, 0xD8, 0x7D, 0x00];
        std::fs::write(dir.0.join(FILE_NAME), &second).unwrap();
        let again = SettingsStore::open(&dir.0);
        assert_eq!(again.get(), RingSettings::default(), "a broken file never turns a transfer on");
        assert_eq!(std::fs::read(dir.0.join("ring.json.corrupt")).unwrap(), first);
        assert_eq!(std::fs::read(dir.0.join("ring.json.corrupt.1")).unwrap(), second);
    }

    /// A file that cannot be read for the moment may be perfectly good: it is not put aside as garbage, and a change does not replace it
    /// (a replace would have worked: the owner's settings would have been gone).
    #[test]
    fn a_settings_file_that_cannot_be_read_is_left_alone_and_a_change_is_refused() {
        let dir = TempDir::new("ring-unreadable");
        let first = SettingsStore::open(&dir.0);
        first.change(&json!({"enabled": true, "ringSeconds": 55})).unwrap();
        let before = std::fs::read(dir.0.join(FILE_NAME)).unwrap();
        let Some(lock) = crate::secret_file::testing::make_unreadable(&dir.0.join(FILE_NAME)) else {
            eprintln!("skipped: this user reads every file");
            return;
        };
        let store = SettingsStore::open(&dir.0);
        assert_eq!(store.get(), RingSettings::default(), "everything stays off while nothing can be read");
        assert!(store.load_problem().is_some_and(|p| p.contains("could not be read")), "{:?}", store.load_problem());
        let e = store.change(&json!({"ringSeconds": 60})).unwrap_err();
        assert!(e.to_string().contains(FILE_NAME), "{e}");
        assert!(!dir.0.join("ring.json.corrupt").exists(), "a file that may be good is not put aside");
        drop(lock);
        assert_eq!(std::fs::read(dir.0.join(FILE_NAME)).unwrap(), before, "and it is as it was");
        assert_eq!(SettingsStore::open(&dir.0).get().ring_seconds, 55, "read again once it can be");
    }

    /// A file another program holds for a moment at start-up (an antivirus scan, an indexer) does not leave the settings off until the next start: it is
    /// read again at once a few times, and if it is still busy everything stays off (never on for want of a read), the owner is told, a change is
    /// refused, and it is read again every so often until it can be.
    #[test]
    fn a_settings_file_that_is_busy_at_start_is_read_again_and_the_settings_come_on_when_it_can_be() {
        use crate::secret_file::Patience;
        use std::time::Duration;
        let ms = Duration::from_millis;
        let dir = TempDir::new("ring-busy");
        SettingsStore::open(&dir.0).change(&json!({"enabled": true, "ringSeconds": 55})).unwrap();
        let file = dir.0.join(FILE_NAME);
        std::fs::rename(&file, dir.0.join("held.json")).unwrap();
        std::fs::create_dir(&file).unwrap(); // a read fails where the file belongs
        let store = SettingsStore::open_patiently(&dir.0, Patience { start: vec![ms(1)], later: vec![ms(20)], window: Duration::ZERO });
        assert_eq!(store.get(), RingSettings::default(), "everything stays off while nothing can be read");
        assert!(store.load_problem().is_some_and(|p| p.contains("could not be read") && p.contains("trying again")), "{:?}", store.load_problem());
        assert!(store.change(&json!({"ringSeconds": 60})).is_err(), "a change is refused meanwhile");
        // The file can be read again: the next look at the settings reads it.
        std::fs::remove_dir(&file).unwrap();
        std::fs::rename(dir.0.join("held.json"), &file).unwrap();
        std::thread::sleep(ms(60));
        let now = store.get();
        assert!(now.enabled && now.ring_seconds == 55, "the owner's settings are in force again: {now:?}");
        assert_eq!(store.load_problem(), None, "and nothing is wrong");
        assert!(store.change(&json!({"ringSeconds": 60})).is_ok(), "a change is kept again");
        assert_eq!(SettingsStore::open(&dir.0).get().ring_seconds, 60);
    }

    /// The waits at the start are for real: a file that is free again within them is read by the open itself.
    #[test]
    fn a_settings_file_that_is_free_again_within_the_short_waits_is_read_by_the_open_itself() {
        use crate::secret_file::Patience;
        use std::time::Duration;
        let ms = Duration::from_millis;
        let dir = TempDir::new("ring-busy-short");
        SettingsStore::open(&dir.0).change(&json!({"enabled": true, "ringSeconds": 45})).unwrap();
        // Held by another program (it cannot be read while it is), and let go a little later.
        let Some(lock) = crate::secret_file::testing::make_unreadable(&dir.0.join(FILE_NAME)) else {
            eprintln!("skipped: this user reads every file");
            return;
        };
        let fixer = std::thread::spawn(move || {
            std::thread::sleep(ms(80));
            drop(lock);
        });
        // (Nothing later would read it again in this test, so what is read is what the open itself read.)
        let store = SettingsStore::open_patiently(&dir.0, Patience { start: vec![ms(30); 20], later: vec![Duration::from_secs(3_600)], window: Duration::ZERO });
        fixer.join().unwrap();
        assert!(store.get().enabled && store.get().ring_seconds == 45, "read by the open itself");
        assert_eq!(store.load_problem(), None);
    }

    #[test]
    fn a_settings_file_that_cannot_be_put_aside_is_not_written_over_and_a_change_says_why() {
        let dir = TempDir::new("ring-protected");
        let bytes: &[u8] = &[b'{', 0xC3, 0x28];
        std::fs::write(dir.0.join(FILE_NAME), bytes).unwrap();
        std::fs::create_dir_all(dir.0.join("ring.json.corrupt")).unwrap();
        for n in 1..40 {
            std::fs::create_dir_all(dir.0.join(format!("ring.json.corrupt.{n}"))).unwrap();
        }
        let store = SettingsStore::open(&dir.0);
        assert_eq!(store.get(), RingSettings::default());
        assert!(store.load_problem().is_some());
        let e = store.change(&json!({"enabled": true})).unwrap_err();
        assert!(e.to_string().contains("ring.json"), "{e}");
        assert!(!store.get().enabled, "a change that could not be kept is not made");
        assert_eq!(std::fs::read(dir.0.join(FILE_NAME)).unwrap(), bytes, "the file is as it was");
    }

    #[test]
    fn a_file_with_transfers_on_and_messages_off_is_read_with_messages_on() {
        let dir = TempDir::new("ring-fallback");
        std::fs::write(dir.0.join(FILE_NAME), r#"{"version": 1, "enabled": true, "takeMessages": false}"#).unwrap();
        let s = SettingsStore::open(&dir.0).get();
        assert!(s.enabled && s.take_messages && s.messages_on());
        // Messages alone are fine.
        let store = SettingsStore::in_memory(RingSettings { take_messages: true, ..RingSettings::default() });
        assert!(store.get().messages_on() && !store.get().enabled);
    }

    #[test]
    fn a_vip_is_known_by_the_last_nine_digits() {
        let s = RingSettings { vip_numbers: vec!["0491 570 006".into()], ..RingSettings::default() };
        assert!(s.is_vip("+61491570006") && s.is_vip("0491570006") && s.is_vip("61 491 570 006"));
        assert!(!s.is_vip("0491 570 156") && !s.is_vip("") && !s.is_vip("Private"));
    }

    #[test]
    fn a_timed_away_ends_by_itself() {
        let s = RingSettings { away: Away::On, away_until: Some(1_000), ..RingSettings::default() };
        assert_eq!(s.away_at(999), Away::On);
        assert_eq!(s.away_at(1_000), Away::Auto);
        assert_eq!(RingSettings { away: Away::On, ..RingSettings::default() }.away_at(u64::MAX), Away::On, "no end: until turned off");
        assert_eq!(RingSettings { away: Away::Off, away_until: Some(1), ..RingSettings::default() }.away_at(5), Away::Off);
    }
}
