//! The setup code (design 4.7.1): how a first owner is made on a server nobody can see.
//!
//! When no `owner.json` exists the server is in setup-only mode for the browser. `oaiy-server auth setup-code`
//! makes a code, on the console, and prints it there: it is never written to the journal (where the `adm` and
//! `systemd-journal` groups can read it) and never to the log ring.
//!
//! - **Format**: 12 characters of Crockford base32 (`0123456789ABCDEFGHJKMNPQRSTVWXYZ`) from 60 random bits,
//!   shown in groups of four (`M6SC-7N75-YR3H`). Input is upper-cased, `O` becomes `0`, `I` and `L` become `1`,
//!   hyphens and spaces are dropped.
//! - **Stored** as `{"v":1,"hash":"<sha256 hex>","expires_ms":N,"wrong_left":100}`, with
//!   `hash = lowercase_hex(SHA-256("oaiy.setup.v1" 0x00 code12))`. A new code replaces any earlier one.
//! - **Limits**: valid 24 hours or until used, and **100 wrong guesses in total**. The counter is in the file, so a
//!   restart does not reset it; at the hundredth the file is deleted and the console must make another. (Five
//!   guesses in total, the first draft's number, let anyone who saw the certificate-transparency entry for the new
//!   host names burn the code with five requests; the code space is 2^60, so a guesser's chance is at most 100 in
//!   2^60, and the address throttle bounds the rest.) Every wrong guess also counts as a failure of the address
//!   in the login throttle (`throttle.rs`).
//! - The code is compared by its hash, in constant time, and is never in a `Debug` form or a log line.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::clock::Clock;
use super::scrub::scrub_line;
use super::store::{FileWriter, Random, StoreError};
use super::token::{self, MintError};

/// A line for the log: through the scrubber, whatever it names.
fn warn(line: String) {
    log::warn!("{}", scrub_line(&line));
}

/// Crockford base32.
pub const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
/// Characters in a code: 12 of 5 bits.
pub const CODE_LEN: usize = 12;
pub const VALID_MS: u64 = 24 * 3_600_000;
/// Wrong guesses in total.
pub const WRONG_GUESSES: u32 = 100;
const HASH_PREFIX: &[u8] = b"oaiy.setup.v1\0";
const FILE_VERSION: u64 = 1;

/// The 12 characters for these random bytes: the first 60 bits, five at a time, most significant first.
pub fn encode(random: [u8; 8]) -> String {
    let bits = u64::from_be_bytes(random);
    (0..CODE_LEN as u32)
        .map(|i| ALPHABET[((bits >> (59 - 5 * i)) & 31) as usize] as char)
        .collect()
}

/// A code as it is shown: `M6SC-7N75-YR3H`.
pub fn display(code12: &str) -> String {
    code12
        .as_bytes()
        .chunks(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join("-")
}

/// What a person typed, as the 12 characters it stands for: upper-cased, `O` as `0`, `I` and `L` as `1`, hyphens
/// and spaces dropped. `None` when what is left is not exactly 12 characters of the alphabet.
pub fn normalise(input: &str) -> Option<String> {
    let mut out = String::with_capacity(CODE_LEN);
    for c in input.chars() {
        let c = match c.to_ascii_uppercase() {
            '-' | ' ' => continue,
            'O' => '0',
            'I' | 'L' => '1',
            other => other,
        };
        if out.len() >= CODE_LEN || !c.is_ascii() || !ALPHABET.contains(&(c as u8)) {
            return None;
        }
        out.push(c);
    }
    (out.len() == CODE_LEN).then_some(out)
}

/// What is stored for a code: `lowercase_hex(SHA-256("oaiy.setup.v1" 0x00 code12))`.
pub fn hash(code12: &str) -> String {
    let mut h = Sha256::new();
    h.update(HASH_PREFIX);
    h.update(code12.as_bytes());
    token::to_hex(&h.finalize())
}

/// What a code file says.
#[derive(Clone, Serialize, Deserialize)]
struct Stored {
    v: u64,
    hash: String,
    expires_ms: u64,
    wrong_left: u32,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

impl std::fmt::Debug for Stored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stored")
            .field("hash", &"[redacted]")
            .field("expires_ms", &self.expires_ms)
            .field("wrong_left", &self.wrong_left)
            .finish()
    }
}

/// What `GET /api/auth/info` says of the code: `active`, `expired` or `none`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Active,
    Expired,
    None,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Active => "active",
            Status::Expired => "expired",
            Status::None => "none",
        }
    }
}

/// What a guess found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// The right code, not expired.
    Valid,
    /// Wrong: this many guesses are left (the code is burned at zero).
    Wrong { attempts_left: u32 },
    /// The code is right or wrong, but it has expired: `410 setup_code_expired`.
    Expired,
    /// There is no code (never made, used, or burned).
    NoCode,
}

/// The code of a data folder.
pub struct SetupCode {
    path: PathBuf,
    clock: Arc<dyn Clock>,
    writer: Arc<dyn FileWriter>,
    state: Mutex<Option<Stored>>,
}

impl SetupCode {
    /// Read `<auth>/setup-code.json`. A file that is not there is no code; one that cannot be read is an error
    /// naming it (never "no code": a permission error must not quietly reopen a burned code's slot), and one
    /// that cannot be parsed is set aside as no code (a mangled file must not stop a server that has an owner).
    pub fn open(
        auth_dir: &Path,
        clock: Arc<dyn Clock>,
        writer: Arc<dyn FileWriter>,
    ) -> Result<SetupCode, StoreError> {
        let path = auth_dir.join("setup-code.json");
        let stored = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<Stored>(&bytes) {
                Ok(s) if s.v == FILE_VERSION => Some(s),
                Ok(s) => {
                    warn(format!(
                        "auth: {} is version {} and this build reads {FILE_VERSION}: ignored",
                        path.display(),
                        s.v
                    ));
                    None
                }
                Err(e) => {
                    warn(format!(
                        "auth: {} cannot be read ({e}): ignored",
                        path.display()
                    ));
                    None
                }
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(source) => return Err(StoreError::Unreadable { file: path, source }),
        };
        Ok(SetupCode {
            path,
            clock,
            writer,
            state: Mutex::new(stored),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Stored>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Make a code, replacing any earlier one. The code is returned as it is shown, and exists nowhere else.
    pub fn make(&self, random: &Random) -> Result<String, MakeError> {
        let mut bytes = [0u8; 8];
        random(&mut bytes).map_err(MakeError::Random)?;
        let code = encode(bytes);
        let stored = Stored {
            v: FILE_VERSION,
            hash: hash(&code),
            expires_ms: self.clock.now_ms().saturating_add(VALID_MS),
            wrong_left: WRONG_GUESSES,
            extra: Map::new(),
        };
        // The file and the memory change together, under the lock, as every count does: a count of the code before
        // this one, written slowly, cannot land after it.
        let mut state = self.lock();
        self.write(&stored).map_err(MakeError::Io)?;
        *state = Some(stored);
        Ok(display(&code))
    }

    fn write(&self, stored: &Stored) -> io::Result<()> {
        let mut text = serde_json::to_string(stored).map_err(io::Error::other)?;
        text.push('\n');
        self.writer.write(&self.path, text.as_bytes())
    }

    fn delete(&self) {
        if let Err(e) = std::fs::remove_file(&self.path) {
            if e.kind() != io::ErrorKind::NotFound {
                warn(format!(
                    "auth: {} could not be removed: {e}",
                    self.path.display()
                ));
            }
        }
    }

    /// `active`, `expired` or `none`.
    pub fn status(&self) -> Status {
        let now = self.clock.now_ms();
        match self.lock().as_ref() {
            None => Status::None,
            Some(s) if now >= s.expires_ms => Status::Expired,
            Some(_) => Status::Active,
        }
    }

    /// The wrong guesses left, if there is a code.
    pub fn attempts_left(&self) -> Option<u32> {
        self.lock().as_ref().map(|s| s.wrong_left)
    }

    /// Judge a guess. A wrong one is counted (and written, so that a restart does not give it back); the
    /// hundredth deletes the code. An expired code is not guessed against: nothing is counted.
    pub fn check(&self, input: &str) -> Check {
        let now = self.clock.now_ms();
        let mut guard = self.lock();
        let Some(stored) = guard.as_mut() else {
            return Check::NoCode;
        };
        if now >= stored.expires_ms {
            return Check::Expired;
        }
        let right = match normalise(input) {
            Some(code) => token::hashes_equal(&hash(&code), &stored.hash),
            // Not shaped like a code: it cannot be one, and it costs a guess like any other.
            None => false,
        };
        if right {
            return Check::Valid;
        }
        stored.wrong_left = stored.wrong_left.saturating_sub(1);
        let left = stored.wrong_left;
        // Written (or deleted) under the lock: two guesses cannot write their counts out of order, and a count
        // written slowly cannot land after a newer one and give a guess back at the next restart.
        if left == 0 {
            *guard = None;
            self.delete();
        } else if let Err(e) = self.write(stored) {
            warn(format!(
                "auth: the setup code's count could not be written: {e}"
            ));
        }
        drop(guard);
        Check::Wrong {
            attempts_left: left,
        }
    }

    /// The code was used: it is gone.
    pub fn consume(&self) {
        let mut state = self.lock();
        *state = None;
        self.delete();
    }
}

/// Why a code was not made.
#[derive(Debug)]
pub enum MakeError {
    Random(MintError),
    Io(io::Error),
}

impl std::fmt::Display for MakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MakeError::Random(e) => write!(f, "{e}"),
            MakeError::Io(e) => write!(f, "the setup code could not be written: {e}"),
        }
    }
}

impl std::error::Error for MakeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::clock::ManualClock;
    use crate::auth::store::SecureWriter;
    use crate::secret_file::testing::TempDir;
    use std::sync::atomic::{AtomicU64, Ordering};

    const T0: u64 = 1_790_000_000_000;
    const HOUR: u64 = 3_600_000;

    /// The design's vector (4.7.1): random bits `a1b2c3d4e5f60718`, the first 60 bits used.
    const VECTOR_BYTES: [u8; 8] = [0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07, 0x18];
    const VECTOR_CODE: &str = "M6SC-7N75-YR3H";
    const VECTOR_HASH: &str = "46e441df40d0928dc7c29c0e4963f346f723fd5caa5b213b82766f7fc25e9e74";

    fn fixed(bytes: [u8; 8]) -> Random {
        Arc::new(move |buf: &mut [u8]| {
            buf.copy_from_slice(&bytes);
            Ok(())
        })
    }

    /// A different code each call.
    fn counting() -> Random {
        let n = AtomicU64::new(1);
        Arc::new(move |buf: &mut [u8]| {
            let v = n
                .fetch_add(1, Ordering::SeqCst)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            buf.copy_from_slice(&v.to_be_bytes());
            Ok(())
        })
    }

    fn open(dir: &TempDir, clock: &Arc<ManualClock>) -> SetupCode {
        SetupCode::open(&dir.0.join("auth"), clock.clone(), Arc::new(SecureWriter)).unwrap()
    }

    fn setup() -> (TempDir, Arc<ManualClock>, SetupCode) {
        let dir = TempDir::new("setup-code");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        let code = open(&dir, &clock);
        (dir, clock, code)
    }

    // The vector ---------------------------------------------------------------------------------

    #[test]
    fn the_known_answer_code_and_hash_of_the_design() {
        let code = encode(VECTOR_BYTES);
        assert_eq!(code, "M6SC7N75YR3H");
        assert_eq!(display(&code), VECTOR_CODE);
        assert_eq!(hash(&code), VECTOR_HASH);
        // Only the first 60 bits are used: the last four of the eight bytes change nothing.
        let mut other = VECTOR_BYTES;
        other[7] = 0x1f;
        assert_eq!(encode(other), code);
        other[7] = 0x28;
        assert_ne!(
            encode(other),
            code,
            "the high half of the last byte is the last four bits of the code"
        );
    }

    #[test]
    fn a_code_is_twelve_characters_of_the_crockford_alphabet_whatever_the_bytes() {
        let mut n = 1u64;
        for _ in 0..2000 {
            n = n
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let code = encode(n.to_be_bytes());
            assert_eq!(code.len(), CODE_LEN);
            assert!(code.bytes().all(|b| ALPHABET.contains(&b)), "{code}");
            assert_eq!(normalise(&display(&code)), Some(code));
        }
        // The alphabet has no I, L, O or U.
        assert!(!ALPHABET.contains(&b'I') && !ALPHABET.contains(&b'L'));
        assert!(!ALPHABET.contains(&b'O') && !ALPHABET.contains(&b'U'));
        assert_eq!(ALPHABET.len(), 32);
        assert_eq!(encode([0; 8]), "000000000000");
        assert_eq!(encode([0xff; 8]), "ZZZZZZZZZZZZ");
    }

    #[test]
    fn what_a_person_types_is_read_the_way_the_design_says() {
        for typed in [
            "M6SC-7N75-YR3H",
            "m6sc-7n75-yr3h",
            "M6SC 7N75 YR3H",
            "m6sc7n75yr3h",
            "  M6SC-7N75-YR3H  ",
            "M-6-S-C-7-N-7-5-Y-R-3-H",
        ] {
            assert_eq!(
                normalise(typed).as_deref(),
                Some("M6SC7N75YR3H"),
                "{typed:?}"
            );
        }
        // O is 0, I and L are 1.
        assert_eq!(normalise("O0O0-O0O0-O0O0").as_deref(), Some("000000000000"));
        assert_eq!(normalise("IlIl-LiLi-1111").as_deref(), Some("111111111111"));
        for bad in [
            "",
            "M6SC-7N75-YR3",
            "M6SC-7N75-YR3HH",
            "M6SC-7N75-YR3U",
            "M6SC-7N75-YR3\u{ff28}",
            "M6SC_7N75_YR3H",
            "M6SC\n7N75YR3H",
        ] {
            assert_eq!(normalise(bad), None, "{bad:?}");
        }
        // A trailing hyphen is dropped like any other.
        assert_eq!(
            normalise("M6SC-7N75-YR3H-").as_deref(),
            Some("M6SC7N75YR3H")
        );
    }

    // Making, checking, counting ------------------------------------------------------------------

    #[test]
    fn a_code_is_made_shown_once_stored_only_as_a_hash_and_accepted_as_typed() {
        let (dir, _clock, code) = setup();
        assert_eq!(code.status(), Status::None);
        let shown = code.make(&fixed(VECTOR_BYTES)).unwrap();
        assert_eq!(shown, VECTOR_CODE);
        assert_eq!(code.status(), Status::Active);
        assert_eq!(code.attempts_left(), Some(100));
        let file = std::fs::read_to_string(dir.0.join("auth").join("setup-code.json")).unwrap();
        let v: Value = serde_json::from_str(&file).unwrap();
        assert_eq!(v["hash"], VECTOR_HASH);
        assert_eq!(v["wrong_left"], 100);
        assert_eq!(v["expires_ms"], T0 + 24 * HOUR);
        assert_eq!(v["v"], 1);
        assert!(
            !file.contains("M6SC") && !file.contains("7N75"),
            "the code is not in the file: {file}"
        );
        assert_eq!(code.check("m6sc 7n75 yr3h"), Check::Valid);
        // A right code stays valid until it is used (it is not consumed by being checked).
        assert_eq!(code.check(VECTOR_CODE), Check::Valid);
        code.consume();
        assert_eq!(code.status(), Status::None);
        assert!(!dir.0.join("auth").join("setup-code.json").exists());
        assert_eq!(code.check(VECTOR_CODE), Check::NoCode);
    }

    #[test]
    fn a_wrong_guess_counts_down_and_the_hundred_and_first_finds_a_burned_code() {
        let (dir, _clock, code) = setup();
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        for i in 1..=99 {
            assert_eq!(
                code.check("AAAA-AAAA-AAAA"),
                Check::Wrong {
                    attempts_left: 100 - i
                },
                "guess {i}"
            );
        }
        assert_eq!(code.status(), Status::Active);
        // The hundredth wrong guess burns it: the file is gone, and even the right code no longer works.
        assert_eq!(
            code.check("AAAA-AAAA-AAAA"),
            Check::Wrong { attempts_left: 0 }
        );
        assert_eq!(code.status(), Status::None);
        assert!(!dir.0.join("auth").join("setup-code.json").exists());
        assert_eq!(code.check("AAAA-AAAA-AAAA"), Check::NoCode, "the 101st");
        assert_eq!(
            code.check(VECTOR_CODE),
            Check::NoCode,
            "burned, not just counted"
        );
    }

    #[test]
    fn a_guess_that_is_not_shaped_like_a_code_costs_a_guess_too() {
        let (_dir, _clock, code) = setup();
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        assert_eq!(code.check("nonsense"), Check::Wrong { attempts_left: 99 });
        assert_eq!(code.check(""), Check::Wrong { attempts_left: 98 });
        assert_eq!(
            code.check(&"9".repeat(10_000)),
            Check::Wrong { attempts_left: 97 }
        );
    }

    #[test]
    fn the_count_survives_a_restart_and_is_not_given_back() {
        let dir = TempDir::new("setup-restart");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        let first = open(&dir, &clock);
        first.make(&fixed(VECTOR_BYTES)).unwrap();
        for _ in 0..40 {
            first.check("AAAA-AAAA-AAAA");
        }
        drop(first);
        // A new process reads the file.
        let second = open(&dir, &clock);
        assert_eq!(second.status(), Status::Active);
        assert_eq!(second.attempts_left(), Some(60));
        assert_eq!(
            second.check("AAAA-AAAA-AAAA"),
            Check::Wrong { attempts_left: 59 }
        );
        drop(second);
        let third = open(&dir, &clock);
        assert_eq!(third.attempts_left(), Some(59));
        assert_eq!(third.check(VECTOR_CODE), Check::Valid);
        // The last wrong guess before a restart burns it for the next process too.
        for _ in 0..59 {
            third.check("AAAA-AAAA-AAAA");
        }
        drop(third);
        let fourth = open(&dir, &clock);
        assert_eq!(fourth.status(), Status::None);
        assert_eq!(fourth.check(VECTOR_CODE), Check::NoCode);
    }

    #[test]
    fn a_code_is_valid_for_twenty_four_hours_and_then_expired_not_none_and_nothing_is_counted() {
        let (_dir, clock, code) = setup();
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        clock.set(T0 + 24 * HOUR - 1);
        assert_eq!(code.status(), Status::Active);
        assert_eq!(code.check(VECTOR_CODE), Check::Valid);
        clock.set(T0 + 24 * HOUR);
        assert_eq!(code.status(), Status::Expired);
        assert_eq!(code.check(VECTOR_CODE), Check::Expired);
        assert_eq!(code.check("AAAA-AAAA-AAAA"), Check::Expired);
        assert_eq!(
            code.attempts_left(),
            Some(100),
            "an expired code is not guessed against"
        );
        // Nothing regenerates it on its own: it stays expired until the console makes another.
        clock.advance(1000 * HOUR);
        assert_eq!(code.status(), Status::Expired);
    }

    #[test]
    fn a_second_code_replaces_the_first_and_its_count() {
        let (dir, _clock, code) = setup();
        let random = counting();
        let first = code.make(&random).unwrap();
        for _ in 0..30 {
            code.check("AAAA-AAAA-AAAA");
        }
        let second = code.make(&random).unwrap();
        assert_ne!(first, second);
        assert_eq!(code.attempts_left(), Some(100));
        assert_eq!(
            code.check(&first),
            Check::Wrong { attempts_left: 99 },
            "the old one is dead"
        );
        assert_eq!(code.check(&second), Check::Valid);
        let v: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.0.join("auth").join("setup-code.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(v["hash"], hash(&normalise(&second).unwrap()));
    }

    #[test]
    fn a_failing_write_makes_no_code_and_a_failing_count_write_does_not_stop_the_count() {
        struct Toggle(std::sync::atomic::AtomicBool);
        impl FileWriter for Toggle {
            fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
                if self.0.load(Ordering::SeqCst) {
                    Err(io::Error::other("disk full"))
                } else {
                    SecureWriter.write(path, bytes)
                }
            }
        }
        let dir = TempDir::new("setup-full");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        let writer = Arc::new(Toggle(std::sync::atomic::AtomicBool::new(true)));
        let code = SetupCode::open(&dir.0.join("auth"), clock, writer.clone()).unwrap();
        assert!(matches!(
            code.make(&fixed(VECTOR_BYTES)),
            Err(MakeError::Io(_))
        ));
        assert_eq!(code.status(), Status::None);
        writer.0.store(false, Ordering::SeqCst);
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        writer.0.store(true, Ordering::SeqCst);
        assert_eq!(
            code.check("AAAA-AAAA-AAAA"),
            Check::Wrong { attempts_left: 99 }
        );
        assert_eq!(code.attempts_left(), Some(99), "memory is authoritative");
    }

    /// A writer for which the write of one particular text is slow.
    struct Slow {
        marker: &'static str,
        wait: std::time::Duration,
    }

    impl FileWriter for Slow {
        fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
            if String::from_utf8_lossy(bytes).contains(self.marker) {
                std::thread::sleep(self.wait);
            }
            SecureWriter.write(path, bytes)
        }
    }

    fn slow_setup(marker: &'static str) -> (TempDir, SetupCode) {
        let dir = TempDir::new("setup-slow");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let writer = Arc::new(Slow {
            marker,
            wait: std::time::Duration::from_millis(300),
        });
        let clock = Arc::new(ManualClock::new(T0));
        let code = SetupCode::open(&dir.0.join("auth"), clock, writer).unwrap();
        (dir, code)
    }

    fn on_disk(dir: &TempDir) -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(dir.0.join("auth").join("setup-code.json")).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn two_wrong_guesses_at_once_leave_the_newer_count_on_disk_however_slow_the_older_write_is() {
        // The count after the first wrong guess (99) is written slowly; the second guess (98) counts and writes while
        // it sleeps. The file must end with 98: an older count landing last would give a guess back at a restart.
        let (dir, code) = slow_setup("\"wrong_left\":99");
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        std::thread::scope(|s| {
            s.spawn(|| code.check("AAAA-AAAA-AAAA"));
            std::thread::sleep(std::time::Duration::from_millis(100));
            s.spawn(|| code.check("BBBB-BBBB-BBBB"));
        });
        assert_eq!(code.attempts_left(), Some(98));
        assert_eq!(
            on_disk(&dir)["wrong_left"],
            98,
            "the file agrees with memory"
        );
    }

    #[test]
    fn forty_wrong_guesses_at_once_are_all_on_disk_and_a_restart_gives_none_back() {
        let dir = TempDir::new("setup-forty");
        std::fs::create_dir_all(dir.0.join("auth")).unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        let code = open(&dir, &clock);
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        std::thread::scope(|s| {
            for i in 0..40 {
                let code = &code;
                s.spawn(move || code.check(&format!("AAAA-AAAA-{i:04}")));
            }
        });
        assert_eq!(code.attempts_left(), Some(60));
        assert_eq!(on_disk(&dir)["wrong_left"], 60);
        assert_eq!(open(&dir, &clock).attempts_left(), Some(60));
    }

    #[test]
    fn a_new_code_is_the_one_on_disk_whatever_an_older_count_write_is_doing() {
        // A count of the first code is being written slowly when the next code is made: the file must end as the
        // next code (an older count landing after it would bring the first code back at a restart).
        let (dir, code) = slow_setup("\"wrong_left\":99");
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        let second = std::thread::scope(|s| {
            s.spawn(|| code.check("AAAA-AAAA-AAAA"));
            std::thread::sleep(std::time::Duration::from_millis(100));
            code.make(&counting()).unwrap()
        });
        assert_eq!(
            on_disk(&dir)["hash"],
            hash(&normalise(&second).unwrap()),
            "the file holds the newer code"
        );
        assert_eq!(on_disk(&dir)["wrong_left"], 100);
    }

    #[test]
    fn a_failure_of_the_random_source_makes_no_code() {
        let (_dir, _clock, code) = setup();
        let broken: Random = Arc::new(|_: &mut [u8]| Err(MintError::NoRandomness));
        assert!(matches!(
            code.make(&broken),
            Err(MakeError::Random(MintError::NoRandomness))
        ));
        assert_eq!(code.status(), Status::None);
    }

    #[test]
    fn a_file_that_cannot_be_read_is_an_error_naming_it_and_one_that_cannot_be_parsed_is_no_code() {
        let dir = TempDir::new("setup-bad");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let clock = Arc::new(ManualClock::new(T0));
        // Unparsable and of another version: no code, no error.
        for text in [
            "not json",
            r#"{"v":2,"hash":"x","expires_ms":1,"wrong_left":1}"#,
            "{}",
        ] {
            std::fs::write(auth.join("setup-code.json"), text).unwrap();
            let code = SetupCode::open(&auth, clock.clone(), Arc::new(SecureWriter)).unwrap();
            assert_eq!(code.status(), Status::None, "{text}");
        }
        // A directory where the file belongs: a read error that is not "not there".
        std::fs::remove_file(auth.join("setup-code.json")).unwrap();
        std::fs::create_dir(auth.join("setup-code.json")).unwrap();
        let err = SetupCode::open(&auth, clock, Arc::new(SecureWriter))
            .err()
            .expect("an error");
        assert!(matches!(err, StoreError::Unreadable { .. }));
        assert!(err.to_string().contains("setup-code.json"), "{err}");
    }

    #[test]
    fn debug_never_shows_the_hash() {
        let (_dir, _clock, code) = setup();
        code.make(&fixed(VECTOR_BYTES)).unwrap();
        let shown = format!("{:?}", code.lock().as_ref().unwrap());
        assert!(!shown.contains(VECTOR_HASH) && !shown.contains("M6SC"));
    }
}
