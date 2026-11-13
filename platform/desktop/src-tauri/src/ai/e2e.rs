//! The sealed envelope a provider's web app talks to this desktop through.
//!
//! One NaCl-compatible `crypto_box` construction — X25519 ECDH → HSalsa20 key
//! derivation → XSalsa20-Poly1305 — which is exactly what the browser's
//! tweetnacl `box` computes, so the two interoperate byte for byte. It is NOT
//! XChaCha20-Poly1305; the two are one word apart in a spec and would silently
//! never open each other's frames. `e2e-envelope-vectors.json`, shared with the
//! provider's own implementation and compiled in below, is what makes that
//! claim checkable rather than asserted.
//!
//! Keys: this desktop holds ONE long-term X25519 identity, minted on first use
//! and kept beside the other on-device state. The peer is a per-request
//! ephemeral keypair the browser never persists, whose public half arrives as
//! routing plaintext on the request row — outside the sealed body, because the
//! shared key cannot be derived without it.
//!
//! Nonce scheme: 24 bytes; byte 0 is the direction, bytes 1..=23 a big-endian
//! counter (a u64 in practice, so bytes 1..16 stay zero). The request envelope
//! is counter 0 and each later frame in the same direction increments. A
//! counter at or below the last seen for a direction is a replay and refused.
//!
//! Session scope is ONE relayed request: counters restart per request and the
//! peer key is pinned at first contact, so a later frame naming the same
//! request under a different key is refused rather than opened.
//!
//! A restarted counter is safe only while no two requests share a box key, so
//! a box gets one request, ever, in every lane and across restarts: see
//! [`SpentBoxes`].
//!
//! Scope note, stated rather than hidden: the reference implementation this is
//! adapted from keeps its identity secret in the OS credential store. OAIY has
//! no credential-store module, so the secret is a file with owner-only
//! permissions where the platform offers them — the same posture as this app's
//! other on-device secrets, and weaker than a keyring on a shared machine.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use base64::Engine as _;
use crypto_box::aead::Aead;
use crypto_box::{PublicKey, SalsaBox, SecretKey};
use sha2::{Digest as _, Sha256};

/// The identity secret's file name, beside the rest of the app's data.
const IDENTITY_KEY_FILE: &str = "desktop-e2e-identity.key";
/// How many boxes this desktop will ever open: eleven years at a thousand requests a day. None is
/// forgotten, because a forgotten box is a replay, and a backend sets the rate: rows of junk would
/// push a recorded request out of any window. At the limit sealed work stops, which is the safe way
/// to be wrong. A new identity starts a new list, as the file's error says.
const MAX_SPENT_BOXES: usize = 1 << 22;
/// What is kept of a box's fingerprint.
const SPENT_BOX_BYTES: usize = 16;
/// A nonce no frame uses (`parse_nonce` knows two direction bytes), for a box to identify itself.
const FINGERPRINT_NONCE: [u8; 24] = [0xFF; 24];

/// browser → desktop (request envelopes and later input frames).
pub const DIR_BROWSER_TO_DESKTOP: u8 = 0x00;
/// desktop → browser (stream frames and the terminal body).
pub const DIR_DESKTOP_TO_BROWSER: u8 = 0x01;

/// A sealed envelope carries at most this much plaintext.
pub const MAX_ENVELOPE_PLAINTEXT_BYTES: usize = 256 * 1024;
/// XSalsa20-Poly1305 adds a 16-byte tag.
const MAX_ENVELOPE_CIPHERTEXT_BYTES: usize = MAX_ENVELOPE_PLAINTEXT_BYTES + 16;
/// Bound on the in-memory session cache; requests expire server-side in minutes.
const MAX_THREAD_SESSIONS: usize = 64;

/// STANDARD base64 WITH padding — not this app's usual URL-safe alphabet.
///
/// The provider validates the published key with a strict standard-alphabet
/// decode and the browser pins the STRING it was given. A url-safe or unpadded
/// encoding here is either rejected outright or pinned in a shape that can
/// never be changed again without every browser reporting a rotated key.
const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

/// A refused envelope.
///
/// Every crypto and wire-format rejection collapses to one code on the wire, so
/// a caller cannot use the difference between "bad tag" and "replayed counter"
/// as an oracle. The specific reason stays in the message, for this side's logs.
#[derive(Debug, Clone)]
pub struct E2eError {
    code: &'static str,
    message: String,
}

impl E2eError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "sealed_envelope_invalid",
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for E2eError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for E2eError {}

/// The boxes this desktop has opened or sealed with, whichever lane it was.
///
/// A sealed reply's nonce is a direction and a counter that starts at 0 for every request, so a box
/// may serve one request. The browser makes a new ephemeral key for each, so two replies never share
/// a box and a nonce, unless the row is replayed. Nothing in a request expires and a session ends
/// with its request, so a backend holding a recorded `ephPub` and envelope could post them again
/// under a new id at any time, and the reply went out at the same key and nonce as the real answer,
/// under a stream cipher: with text it chose or a second answer, the real one falls to an XOR. The
/// sealed flows lane was open to the same thing, and shares this identity.
///
/// So this keeps boxes, not keys. More than one `ephPub` string makes one box: X25519 ignores the top
/// bit, and a key plus a point of small order gives the same result to any secret that clears that
/// order, which is about half of all identities. The list is by what the box itself does, so every
/// spelling is one entry, and it outlives a restart because a recorded request does.
///
/// A row offered again after a crash is refused too. It has to be: answering it would reuse the
/// nonce just the same. The browser mints a new key when the person tries again.
struct SpentBoxes {
    file: Option<PathBuf>,
    limit: usize,
    state: Mutex<SpentState>,
}

#[derive(Default)]
struct SpentState {
    known: HashSet<[u8; SPENT_BOX_BYTES]>,
    /// Held for the life of the process, because the list is read once. A server on the same data
    /// folder as the window would hold a list of its own, and both would answer one row.
    lock: Option<std::fs::File>,
    read: bool,
}

impl SpentBoxes {
    fn in_memory() -> Self {
        Self { file: None, limit: MAX_SPENT_BOXES, state: Mutex::new(SpentState::default()) }
    }

    /// The one list for a data folder, whichever lane asks: the lanes share an identity.
    fn for_dir(dir: &Path) -> Arc<Self> {
        static OPEN: OnceLock<Mutex<HashMap<PathBuf, Arc<SpentBoxes>>>> = OnceLock::new();
        let mut open = OPEN.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        // Beside the identity, whose boxes they are, and left out of backups with it.
        open.entry(dir.to_path_buf()).or_insert_with(|| Arc::new(Self::load(dir.join("desktop-e2e-spent.bin")))).clone()
    }

    /// Nothing is touched until a request comes: the list is read once this process holds it.
    fn load(path: PathBuf) -> Self {
        Self { file: Some(path), limit: MAX_SPENT_BOXES, state: Mutex::new(SpentState::default()) }
    }

    fn hold(path: &Path) -> Result<std::fs::File, String> {
        let folder = path.parent().unwrap_or(Path::new("."));
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(folder.join("desktop-e2e-spent.lock"))
            .map_err(|e| format!("could not open the spent boxes' lock: {e}"))?;
        fs2::FileExt::try_lock_exclusive(&lock)
            .map_err(|_| "another OAIY process is serving sealed requests from this data folder".to_string())?;
        Ok(lock)
    }

    fn read(path: &Path) -> Result<HashSet<[u8; SPENT_BOX_BYTES]>, String> {
        let mut bytes = Vec::new();
        match std::fs::File::open(path).and_then(|mut file| file.read_to_end(&mut bytes)) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("could not read the spent boxes: {e}")),
        }
        // A crash can leave part of the last entry. The whole ones count, and `append` cuts the rest.
        Ok(bytes
            .chunks_exact(SPENT_BOX_BYTES)
            .map(|entry| {
                let mut known = [0u8; SPENT_BOX_BYTES];
                known.copy_from_slice(entry);
                known
            })
            .collect())
    }

    /// The list, in this process's hands. A failure is tried again at the next request: it may have
    /// been a moment's trouble, a scanner holding the file or a full disk.
    fn ready(path: &Path, state: &mut SpentState) -> Result<(), String> {
        if state.lock.is_none() {
            state.lock = Some(Self::hold(path)?);
        }
        if !state.read {
            state.known = Self::read(path)?;
            state.read = true;
        }
        Ok(())
    }

    /// Have the list in this process's hands, or say why it cannot be.
    fn hold_now(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.file.as_ref().map_or(Ok(()), |path| Self::ready(path, &mut state))
    }

    /// Mark a box used, durably, before anything is sealed with it. False: it was used before.
    fn spend(&self, entry: [u8; SPENT_BOX_BYTES]) -> Result<bool, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(path) = &self.file {
            Self::ready(path, &mut state).map_err(|why| {
                log::error!("a sealed request was refused: {why}");
                why
            })?;
        }
        if state.known.contains(&entry) {
            return Ok(false);
        }
        if state.known.len() >= self.limit {
            let why = "this desktop has opened as many sealed requests as its identity may: it needs a new one \
                       (move desktop-e2e-identity.key and desktop-e2e-spent.bin aside), and each browser that \
                       trusted this one will refuse the new key until its pin is cleared";
            log::error!("{why}");
            return Err(why.into());
        }
        if let Some(path) = &self.file {
            if let Err(why) = Self::append(path, &entry) {
                log::error!("a sealed request was refused: {why}");
                return Err(why);
            }
        }
        state.known.insert(entry);
        Ok(true)
    }

    /// One entry, on the disk before this returns. Past a write the disk cut short, not after it: an
    /// entry left in pieces would have every later one read out of step at the next start.
    fn append(path: &Path, entry: &[u8]) -> Result<(), String> {
        let failed = |e: std::io::Error| format!("could not note the request: {e}");
        let mut file = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(path).map_err(failed)?;
        let length = file.metadata().map_err(failed)?.len();
        let whole = length - length % SPENT_BOX_BYTES as u64;
        if whole != length {
            file.set_len(whole).map_err(failed)?;
        }
        file.seek(std::io::SeekFrom::Start(whole)).map_err(failed)?;
        file.write_all(entry).and_then(|()| file.sync_data()).map_err(failed)
    }
}

/// Let go of the spent list of the folder `dir`, as a process does by ending: it is held for the life of the process,
/// and Windows removes no open file, so a test's folder would stay.
#[cfg(test)]
pub(crate) fn let_go_of_spent_list(dir: &Path) {
    let spent = SpentBoxes::for_dir(dir);
    *spent.state.lock().unwrap_or_else(|e| e.into_inner()) = SpentState::default();
}

/// This desktop's long-term X25519 identity.
pub struct E2eIdentity {
    secret: SecretKey,
    public: PublicKey,
    spent: Arc<SpentBoxes>,
}

impl E2eIdentity {
    /// Build an identity from raw secret bytes (loader and test seam).
    pub fn from_secret_bytes(bytes: [u8; 32]) -> Self {
        let secret = SecretKey::from(bytes);
        let public = secret.public_key();
        Self { secret, public, spent: Arc::new(SpentBoxes::in_memory()) }
    }

    /// Whether this process can open sealed requests now, for a lane to ask before it polls: a row
    /// it claims and cannot open ends as failed, where the process that holds the list would have
    /// answered it. The lock is taken here if it is free, so the question settles who serves.
    pub fn may_open_requests(&self) -> Result<(), String> {
        self.spent.hold_now()
    }

    /// Mark the box a request key makes as used, before anything is sealed with it. False means it
    /// has been, under this spelling or another, and nothing may be sealed with it ([`SpentBoxes`]).
    pub fn spend_request_key(&self, eph_pub: &[u8; 32]) -> Result<bool, E2eError> {
        self.spend_box(&SalsaBox::new(&PublicKey::from(*eph_pub), &self.secret))
    }

    fn spend_box(&self, shared: &SalsaBox) -> Result<bool, E2eError> {
        self.spent.spend(box_fingerprint(shared)?).map_err(E2eError::invalid)
    }

    /// Mint a fresh identity from OS randomness.
    pub fn generate() -> Result<Self, String> {
        let mut bytes = [0u8; 32];
        getrandom::getrandom(&mut bytes)
            .map_err(|e| format!("could not read OS randomness for the e2e identity: {e}"))?;
        Ok(Self::from_secret_bytes(bytes))
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.public.to_bytes()
    }

    /// The published form: standard base64 with padding, 44 characters.
    pub fn public_key_b64(&self) -> String {
        B64.encode(self.public.to_bytes())
    }

    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }

    fn secret_key_b64(&self) -> String {
        B64.encode(self.secret.to_bytes())
    }

    /// Load this machine's identity, minting and persisting one on first use.
    ///
    /// A key that exists but cannot be read is a HARD failure rather than a
    /// quiet re-mint: every browser has pinned the old public key, so silently
    /// rotating would show every user a "key changed" warning they cannot
    /// distinguish from an attack — once per boot, forever.
    pub fn load_or_create(dir: &Path) -> Result<Self, String> {
        // AI and flow workers start independently but must publish/use one key.
        static LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOAD.lock().map_err(|_| "identity initialization was interrupted")?;
        let path: PathBuf = dir.join(IDENTITY_KEY_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let mut identity = Self::decode_secret(text.trim()).ok_or_else(|| {
                    format!(
                        "the e2e identity at {} is unreadable — move it aside to mint a new one, \
                         but every browser that trusted this desktop will ask you to trust it again",
                        path.display()
                    )
                })?;
                identity.spent = SpentBoxes::for_dir(dir);
                return Ok(identity);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("could not read the e2e identity: {e}")),
        }

        let mut identity = Self::generate()?;
        identity.spent = SpentBoxes::for_dir(dir);
        // Written to a staging file and renamed, so an interrupted first boot
        // cannot leave a half-written key that the branch above then refuses. The
        // staging file is created owner-only, not narrowed after the write: a copy
        // readable by everyone existed until then.
        crate::secret_file::write(&path, identity.secret_key_b64())
            .map_err(|e| format!("could not persist the e2e identity: {e}"))?;
        Ok(identity)
    }

    fn decode_secret(b64: &str) -> Option<Self> {
        let bytes = B64.decode(b64).ok()?;
        let bytes: [u8; 32] = bytes.as_slice().try_into().ok()?;
        Some(Self::from_secret_bytes(bytes))
    }
}

/// What tells one box from another: its own output at a nonce no frame uses, so it is the box and not
/// any spelling of a key, hashed so the file holds nothing a box ever sealed.
fn box_fingerprint(shared: &SalsaBox) -> Result<[u8; SPENT_BOX_BYTES], E2eError> {
    let sealed = shared
        .encrypt((&FINGERPRINT_NONCE).into(), [0u8; 16].as_slice())
        .map_err(|_| E2eError::invalid("encryption failed"))?;
    let mut entry = [0u8; SPENT_BOX_BYTES];
    entry.copy_from_slice(&Sha256::digest(&sealed)[..SPENT_BOX_BYTES]);
    Ok(entry)
}

/// The 24-byte nonce for one frame: direction, then a big-endian counter.
pub fn frame_nonce(direction: u8, counter: u64) -> [u8; 24] {
    let mut nonce = [0u8; 24];
    nonce[0] = direction;
    nonce[16..24].copy_from_slice(&counter.to_be_bytes());
    nonce
}

/// Read a frame nonce for an expected direction, returning its counter.
fn parse_nonce(direction: u8, nonce: &[u8; 24]) -> Result<u64, E2eError> {
    if nonce[0] != direction {
        return Err(E2eError::invalid(format!(
            "wrong direction byte {:#04x} (expected {:#04x})",
            nonce[0], direction
        )));
    }
    if nonce[1..16].iter().any(|b| *b != 0) {
        return Err(E2eError::invalid("nonce counter does not fit in u64"));
    }
    let mut be = [0u8; 8];
    be.copy_from_slice(&nonce[16..24]);
    Ok(u64::from_be_bytes(be))
}

/// Decode a fixed-size base64 field (public keys are 32 bytes, nonces 24).
fn decode_b64_fixed<const N: usize>(value: &str, what: &str) -> Result<[u8; N], E2eError> {
    let bytes = B64
        .decode(value.trim())
        .map_err(|_| E2eError::invalid(format!("{what} is not valid base64")))?;
    bytes
        .as_slice()
        .try_into()
        .map_err(|_| E2eError::invalid(format!("{what} must be {N} bytes")))
}

/// One request's session: the shared box plus its replay state.
struct ThreadSession {
    peer_eph_pub: [u8; 32],
    shared: SalsaBox,
    /// Highest inbound counter accepted so far.
    last_inbound: Option<u64>,
    /// Next outbound counter to seal with.
    next_outbound: u64,
}

/// Per-request session cache. Bounded FIFO; entries also drop at completion.
pub struct E2eSessions {
    inner: Mutex<(HashMap<String, ThreadSession>, VecDeque<String>)>,
}

impl Default for E2eSessions {
    fn default() -> Self {
        Self::new()
    }
}

impl E2eSessions {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new((HashMap::new(), VecDeque::new())),
        }
    }

    pub fn thread_count(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0.len()
    }

    /// Drop a request's session once it is finished.
    pub fn drop_thread(&self, thread: &str) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.0.remove(thread).is_some() {
            if let Some(pos) = g.1.iter().position(|k| k == thread) {
                g.1.remove(pos);
            }
        }
    }

    fn insert(&self, thread: &str, session: ThreadSession) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.0.insert(thread.to_string(), session).is_none() {
            g.1.push_back(thread.to_string());
        }
        while g.1.len() > MAX_THREAD_SESSIONS {
            if let Some(old) = g.1.pop_front() {
                g.0.remove(&old);
            }
        }
    }

    /// Open a sealed inbound envelope for `thread`, creating the session on
    /// first contact and enforcing the pinned peer key, the direction and a
    /// strictly increasing counter.
    ///
    /// `envelope_b64` is the wire form: `base64(nonce(24) || ciphertext)`.
    pub fn open_inbound(
        &self,
        identity: &E2eIdentity,
        thread: &str,
        eph_pub_b64: &str,
        envelope_b64: &str,
    ) -> Result<Vec<u8>, E2eError> {
        let eph_pub_bytes = decode_b64_fixed::<32>(eph_pub_b64, "ephPub")?;
        if eph_pub_bytes.iter().all(|b| *b == 0) {
            // A small-order point forces a shared secret anyone can compute.
            // Refuse it rather than "decrypt" under a key that is not secret.
            return Err(E2eError::invalid("ephPub is a small-order point"));
        }
        let (nonce, ct) = unpack_envelope(envelope_b64)?;
        let counter = parse_nonce(DIR_BROWSER_TO_DESKTOP, &nonce)?;

        {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(session) = g.0.get_mut(thread) {
                if session.peer_eph_pub != eph_pub_bytes {
                    return Err(E2eError::invalid(
                        "ephPub changed mid-request (pinned at first contact)",
                    ));
                }
                if session.last_inbound.is_some_and(|last| counter <= last) {
                    return Err(E2eError::invalid(format!(
                        "replayed or out-of-order counter {counter}"
                    )));
                }
                let plaintext = session
                    .shared
                    .decrypt((&nonce).into(), ct.as_slice())
                    .map_err(|_| E2eError::invalid("authentication failed"))?;
                session.last_inbound = Some(counter);
                return Ok(plaintext);
            }
        }

        // First contact must be the counter-0 request envelope; a later counter
        // with no session behind it cannot be checked for replay at all.
        if counter != 0 {
            return Err(E2eError::invalid(
                "the first frame for a request must use counter 0",
            ));
        }
        let peer = PublicKey::from(eph_pub_bytes);
        let shared = SalsaBox::new(&peer, &identity.secret);
        let plaintext = shared
            .decrypt((&nonce).into(), ct.as_slice())
            .map_err(|_| E2eError::invalid("authentication failed"))?;
        if !identity.spend_box(&shared)? {
            return Err(E2eError::invalid("this request's key has been used before"));
        }
        self.insert(
            thread,
            ThreadSession {
                peer_eph_pub: eph_pub_bytes,
                shared,
                last_inbound: Some(0),
                next_outbound: 0,
            },
        );
        Ok(plaintext)
    }

    /// Seal one outbound frame, handing out the next counter for the direction.
    pub fn seal_outbound(&self, thread: &str, plaintext: &[u8]) -> Result<String, E2eError> {
        if plaintext.len() > MAX_ENVELOPE_PLAINTEXT_BYTES {
            return Err(E2eError::invalid("frame exceeds its plaintext cap"));
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let session =
            g.0.get_mut(thread)
                .ok_or_else(|| E2eError::invalid("no open session for this request"))?;
        let nonce = frame_nonce(DIR_DESKTOP_TO_BROWSER, session.next_outbound);
        let ct = session
            .shared
            .encrypt((&nonce).into(), plaintext)
            .map_err(|_| E2eError::invalid("encryption failed"))?;
        session.next_outbound += 1;
        Ok(pack_envelope(&nonce, &ct))
    }
}

/// Pack one wire envelope: `base64(nonce(24) || ciphertext)`.
///
/// A binary concatenation, NOT a JSON object of the two parts. The browser
/// decodes the whole string and slices the first 24 bytes off.
pub fn pack_envelope(nonce: &[u8; 24], ct: &[u8]) -> String {
    let mut bytes = Vec::with_capacity(24 + ct.len());
    bytes.extend_from_slice(nonce);
    bytes.extend_from_slice(ct);
    B64.encode(bytes)
}

/// Split a wire envelope back into `(nonce, ciphertext)`.
pub fn unpack_envelope(envelope_b64: &str) -> Result<([u8; 24], Vec<u8>), E2eError> {
    let bytes = B64
        .decode(envelope_b64.trim())
        .map_err(|_| E2eError::invalid("envelope is not valid base64"))?;
    if bytes.len() < 24 + 16 {
        return Err(E2eError::invalid("envelope is shorter than nonce + tag"));
    }
    if bytes.len() > 24 + MAX_ENVELOPE_CIPHERTEXT_BYTES {
        return Err(E2eError::invalid("envelope exceeds the plaintext cap"));
    }
    let nonce: [u8; 24] = bytes[..24]
        .try_into()
        .map_err(|_| E2eError::invalid("envelope nonce is malformed"))?;
    Ok((nonce, bytes[24..].to_vec()))
}

/// Seal one frame with an explicit key pair and counter, outside any session.
///
/// Used for the terminal error frame when the request envelope never opened:
/// there is no session to count from, but the requester can still be told why.
pub fn seal_detached_envelope(
    own_secret: &[u8; 32],
    peer_public: &[u8; 32],
    direction: u8,
    counter: u64,
    plaintext: &[u8],
) -> Result<String, E2eError> {
    let secret = SecretKey::from(*own_secret);
    let peer = PublicKey::from(*peer_public);
    let shared = SalsaBox::new(&peer, &secret);
    let nonce = frame_nonce(direction, counter);
    let ct = shared
        .encrypt((&nonce).into(), plaintext)
        .map_err(|_| E2eError::invalid("encryption failed"))?;
    Ok(pack_envelope(&nonce, &ct))
}

/// Open one detached frame with an explicit key pair (the interop vectors and
/// the wrong-key tests). No replay state — that lives in [`E2eSessions`].
pub fn open_detached(
    own_secret: &[u8; 32],
    peer_public: &[u8; 32],
    direction: u8,
    nonce_b64: &str,
    ct_b64: &str,
) -> Result<Vec<u8>, E2eError> {
    let nonce = decode_b64_fixed::<24>(nonce_b64, "nonce")?;
    parse_nonce(direction, &nonce)?;
    let ct = B64
        .decode(ct_b64.trim())
        .map_err(|_| E2eError::invalid("ciphertext is not valid base64"))?;
    if ct.len() < 16 || ct.len() > MAX_ENVELOPE_CIPHERTEXT_BYTES {
        return Err(E2eError::invalid("ciphertext length is out of bounds"));
    }
    let secret = SecretKey::from(*own_secret);
    let peer = PublicKey::from(*peer_public);
    let shared = SalsaBox::new(&peer, &secret);
    shared
        .decrypt((&nonce).into(), ct.as_slice())
        .map_err(|_| E2eError::invalid("authentication failed"))
}

/// Seal one detached frame, returning `(nonce_b64, ciphertext_b64)` — the
/// component form the shared vectors record alongside the packed envelope.
/// Only the vectors need the parts separately; everything else packs.
#[cfg(test)]
fn seal_detached_parts(
    own_secret: &[u8; 32],
    peer_public: &[u8; 32],
    direction: u8,
    counter: u64,
    plaintext: &[u8],
) -> Result<(String, String), E2eError> {
    let secret = SecretKey::from(*own_secret);
    let peer = PublicKey::from(*peer_public);
    let shared = SalsaBox::new(&peer, &secret);
    let nonce = frame_nonce(direction, counter);
    let ct = shared
        .encrypt((&nonce).into(), plaintext)
        .map_err(|_| E2eError::invalid("encryption failed"))?;
    Ok((B64.encode(nonce), B64.encode(ct)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPENT_KEYS_FILE: &str = "desktop-e2e-spent.bin";

    fn identity(seed: u8) -> E2eIdentity {
        E2eIdentity::from_secret_bytes([seed; 32])
    }

    fn browser_keypair(seed: u8) -> ([u8; 32], [u8; 32]) {
        let secret = [seed; 32];
        let public = SecretKey::from(secret).public_key().to_bytes();
        (secret, public)
    }

    fn pack_pair(nonce_b64: &str, ct_b64: &str) -> String {
        let nonce = decode_b64_fixed::<24>(nonce_b64, "nonce").unwrap();
        let ct = B64.decode(ct_b64).unwrap();
        pack_envelope(&nonce, &ct)
    }

    /// The vectors the provider's own implementation seals and opens, compiled
    /// in rather than read at runtime so this test cannot quietly skip.
    const VECTORS: &str = include_str!("e2e-envelope-vectors.json");

    #[test]
    fn the_shared_interop_vectors_match_byte_for_byte() {
        // The whole interop claim in one test. NaCl box is XSalsa20-Poly1305;
        // reach for XChaCha20 — one word away in any spec — and every seal and
        // open still SUCCEEDS locally while the browser can open none of it.
        let doc: serde_json::Value = serde_json::from_str(VECTORS).expect("vectors parse");
        let desktop_secret = decode_b64_fixed::<32>(
            doc["desktop"]["secretKey"].as_str().unwrap(),
            "desktop secret",
        )
        .unwrap();
        let browser_secret = decode_b64_fixed::<32>(
            doc["browser"]["secretKey"].as_str().unwrap(),
            "browser secret",
        )
        .unwrap();
        let desktop_public = SecretKey::from(desktop_secret).public_key().to_bytes();
        let browser_public = SecretKey::from(browser_secret).public_key().to_bytes();

        // The recorded public keys must be the ones these secrets derive, or
        // the vectors describe a different pair of peers than we are testing.
        assert_eq!(doc["desktop"]["publicKey"].as_str().unwrap(), B64.encode(desktop_public));
        assert_eq!(doc["browser"]["publicKey"].as_str().unwrap(), B64.encode(browser_public));

        let vectors = doc["vectors"].as_array().expect("vectors array");
        assert_eq!(vectors.len(), 5, "the shared file must not shrink silently");
        for vector in vectors {
            let name = vector["name"].as_str().unwrap();
            let direction = vector["direction"].as_u64().unwrap() as u8;
            let counter = vector["counter"].as_u64().unwrap();
            let nonce = vector["nonce"].as_str().unwrap();
            let plaintext = vector["plaintext"].as_str().unwrap();
            let ciphertext = vector["ciphertext"].as_str().unwrap();

            let (own, peer) = if direction == DIR_BROWSER_TO_DESKTOP {
                (browser_secret, desktop_public)
            } else {
                (desktop_secret, browser_public)
            };
            // Seal here: the bytes must be the recorded ones exactly.
            let (sealed_nonce, sealed_ct) =
                seal_detached_parts(&own, &peer, direction, counter, plaintext.as_bytes())
                    .unwrap_or_else(|e| panic!("seal {name}: {e}"));
            assert_eq!(sealed_nonce, nonce, "nonce mismatch on {name}");
            assert_eq!(sealed_ct, ciphertext, "ciphertext mismatch on {name}");
            // …and the packed wire form must match too: a JSON-object envelope
            // would pass every test above and be unreadable to the browser.
            assert_eq!(
                vector["envelope"].as_str().unwrap(),
                pack_pair(nonce, ciphertext),
                "wire envelope mismatch on {name}"
            );
            // Open from the recipient's side.
            let (recipient, sender_pub) = if direction == DIR_BROWSER_TO_DESKTOP {
                (desktop_secret, browser_public)
            } else {
                (browser_secret, desktop_public)
            };
            let opened = open_detached(&recipient, &sender_pub, direction, nonce, ciphertext)
                .unwrap_or_else(|e| panic!("open {name}: {e}"));
            assert_eq!(String::from_utf8(opened).unwrap(), plaintext, "plaintext on {name}");
        }
    }

    #[test]
    fn the_published_key_is_standard_base64_with_padding() {
        // The provider decodes it strictly and the browser pins the STRING.
        // This app's usual base64 is URL-safe and unpadded, which is either
        // rejected on publish or pinned in a shape that can never change.
        for seed in [0x01u8, 0x7f, 0xfe] {
            let published = identity(seed).public_key_b64();
            assert_eq!(published.len(), 44, "{published}");
            assert!(published.ends_with('='), "padding is required: {published}");
            assert!(
                !published.contains('-') && !published.contains('_'),
                "url-safe alphabet would be refused: {published}"
            );
            assert_eq!(B64.decode(&published).unwrap().len(), 32);
        }
    }

    #[test]
    fn the_nonce_is_a_direction_byte_and_a_big_endian_counter() {
        let nonce = frame_nonce(DIR_BROWSER_TO_DESKTOP, 0x0102_0304_0506_0708);
        assert_eq!(nonce[0], DIR_BROWSER_TO_DESKTOP);
        assert_eq!(&nonce[1..16], &[0u8; 15]);
        assert_eq!(&nonce[16..24], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            parse_nonce(DIR_BROWSER_TO_DESKTOP, &nonce).unwrap(),
            0x0102_0304_0506_0708
        );
        // The wrong direction is refused, and so is a counter too big for u64.
        assert!(parse_nonce(DIR_DESKTOP_TO_BROWSER, &nonce).is_err());
        let mut huge = nonce;
        huge[1] = 1;
        assert!(parse_nonce(DIR_BROWSER_TO_DESKTOP, &huge).is_err());
    }

    #[test]
    fn a_request_round_trips_and_outbound_counters_climb() {
        let desktop = identity(0xD1);
        let (browser_secret, browser_public) = browser_keypair(0xB1);
        let sessions = E2eSessions::new();
        let eph = B64.encode(browser_public);

        let envelope = seal_detached_envelope(
            &browser_secret,
            &desktop.public_key_bytes(),
            DIR_BROWSER_TO_DESKTOP,
            0,
            br#"{"v":1,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let opened = sessions
            .open_inbound(&desktop, "req-1", &eph, &envelope)
            .unwrap();
        assert!(String::from_utf8(opened).unwrap().contains("\"hi\""));

        let first = sessions.seal_outbound("req-1", br#"{"v":1,"kind":"delta"}"#).unwrap();
        let second = sessions.seal_outbound("req-1", br#"{"v":1,"kind":"final"}"#).unwrap();
        let (n0, _) = unpack_envelope(&first).unwrap();
        let (n1, _) = unpack_envelope(&second).unwrap();
        assert_eq!(parse_nonce(DIR_DESKTOP_TO_BROWSER, &n0).unwrap(), 0);
        assert_eq!(parse_nonce(DIR_DESKTOP_TO_BROWSER, &n1).unwrap(), 1);
        // The browser must be able to open both with its own key.
        let (_, ct1) = unpack_envelope(&second).unwrap();
        assert_eq!(
            open_detached(
                &browser_secret,
                &desktop.public_key_bytes(),
                DIR_DESKTOP_TO_BROWSER,
                &B64.encode(n1),
                &B64.encode(ct1),
            )
            .unwrap(),
            br#"{"v":1,"kind":"final"}"#
        );
    }

    #[test]
    fn replays_key_changes_and_wrong_keys_are_all_refused() {
        let desktop = identity(0xD3);
        let (browser_secret, browser_public) = browser_keypair(0xB3);
        let eph = B64.encode(browser_public);
        let sessions = E2eSessions::new();

        let seal = |counter: u64, body: &[u8]| {
            seal_detached_envelope(
                &browser_secret,
                &desktop.public_key_bytes(),
                DIR_BROWSER_TO_DESKTOP,
                counter,
                body,
            )
            .unwrap()
        };

        let first = seal(0, b"turn 1");
        sessions.open_inbound(&desktop, "req-3", &eph, &first).unwrap();
        // The same frame again is a replay.
        assert!(sessions.open_inbound(&desktop, "req-3", &eph, &first).is_err());
        // Counter 1 is fine once, then not.
        let second = seal(1, b"input");
        sessions.open_inbound(&desktop, "req-3", &eph, &second).unwrap();
        assert!(sessions.open_inbound(&desktop, "req-3", &eph, &second).is_err());
        // A first frame that is not counter 0 has nothing to be checked against.
        assert!(sessions.open_inbound(&desktop, "req-late", &eph, &seal(5, b"late")).is_err());
        // The peer key is pinned for the life of the request.
        let (_, other_public) = browser_keypair(0x44);
        assert!(sessions
            .open_inbound(&desktop, "req-3", &B64.encode(other_public), &seal(2, b"third"))
            .is_err());
        // An all-zero peer key is refused before any decryption happens.
        assert!(sessions
            .open_inbound(&desktop, "req-zero", &B64.encode([0u8; 32]), &seal(0, b"x"))
            .is_err());
        // A different desktop identity cannot open it at all.
        assert_eq!(
            sessions
                .open_inbound(&identity(0x99), "req-other", &eph, &seal(0, b"secret"))
                .unwrap_err()
                .code(),
            "sealed_envelope_invalid"
        );
    }

    /// A recorded request and the frames it was answered with, for the replays below.
    fn sealed_request(desktop: &E2eIdentity, seed: u8) -> (String, String) {
        let (secret, public) = browser_keypair(seed);
        let envelope = seal_detached_envelope(&secret, &desktop.public_key_bytes(), DIR_BROWSER_TO_DESKTOP, 0, b"{\"v\":1}").unwrap();
        (B64.encode(public), envelope)
    }

    #[test]
    fn a_recorded_request_opens_once_under_any_id_in_any_lane() {
        let desktop = identity(0xD4);
        let (eph, envelope) = sealed_request(&desktop, 0xB4);
        let chat = E2eSessions::new();
        chat.open_inbound(&desktop, "req-1", &eph, &envelope).unwrap();
        let first = chat.seal_outbound("req-1", b"the answer").unwrap();
        chat.drop_thread("req-1");

        // The backend posts the same row under a new id, or the old one once its session is gone.
        for id in ["req-2", "req-1"] {
            let refused = chat.open_inbound(&desktop, id, &eph, &envelope).unwrap_err();
            assert_eq!(refused.code(), "sealed_envelope_invalid", "{id}");
            assert!(refused.message().contains("used before"), "{}", refused.message());
            assert!(chat.seal_outbound(id, b"chosen").is_err(), "no session, so nothing to seal with");
        }
        // The flows lane makes a set of sessions for each poll, and shares the identity.
        assert!(E2eSessions::new().open_inbound(&desktop, "flow-1", &eph, &envelope).is_err());
        assert!(!first.is_empty());
    }

    #[test]
    fn flipping_a_keys_top_bit_is_not_a_new_key() {
        // X25519 ignores that bit, so this string opens the same box.
        let desktop = identity(0xD5);
        let (eph, envelope) = sealed_request(&desktop, 0xB5);
        let sessions = E2eSessions::new();
        sessions.open_inbound(&desktop, "req-1", &eph, &envelope).unwrap();
        let mut flipped = decode_b64_fixed::<32>(&eph, "ephPub").unwrap();
        flipped[31] ^= 0x80;
        let refused = sessions.open_inbound(&desktop, "req-2", &B64.encode(flipped), &envelope).unwrap_err();
        assert!(refused.message().contains("used before"), "the box opened: {}", refused.message());
    }

    #[test]
    fn the_later_frames_of_a_live_request_are_not_replays() {
        let desktop = identity(0xD6);
        let (secret, public) = browser_keypair(0xB6);
        let eph = B64.encode(public);
        let seal = |counter| seal_detached_envelope(&secret, &desktop.public_key_bytes(), DIR_BROWSER_TO_DESKTOP, counter, b"input").unwrap();
        let sessions = E2eSessions::new();
        sessions.open_inbound(&desktop, "req-1", &eph, &seal(0)).unwrap();
        sessions.open_inbound(&desktop, "req-1", &eph, &seal(1)).unwrap();
        sessions.open_inbound(&desktop, "req-1", &eph, &seal(2)).unwrap();
    }

    /// A list that was dropped has let go of its folder's lock, in this process. The lock belongs to the open file,
    /// though, and a child another test is starting at that moment holds a copy of it until it runs its own
    /// program: a list opened the instant the last one went can find the lock held for a few milliseconds more, and
    /// answer "another OAIY process". A test that opens the folder again waits here for the lock to be free first.
    fn released(path: &Path) {
        let lock = path.parent().unwrap().join("desktop-e2e-spent.lock");
        let give_up = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            // No lock file: nothing ever held it.
            let Ok(file) = std::fs::OpenOptions::new().write(true).open(&lock) else { return };
            if fs2::FileExt::try_lock_exclusive(&file).is_ok() {
                // Let go of it by name, not only by closing: a copy of this file in a child would hold it on.
                let _ = fs2::FileExt::unlock(&file);
                return;
            }
            assert!(std::time::Instant::now() < give_up, "the dropped list's lock is still held after ten seconds");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn spent_boxes_outlive_a_restart_and_a_torn_write() {
        let dir = crate::secret_file::testing::TempDir::new("e2espent");
        let path = dir.0.join(SPENT_KEYS_FILE);
        let store = SpentBoxes::load(path.clone());
        assert!(store.spend([1u8; 16]).unwrap());
        assert!(!store.spend([1u8; 16]).unwrap());
        assert!(store.spend([2u8; 16]).unwrap());
        drop(store);
        released(&path);

        // Part of a third entry, as a crash mid-write leaves.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[9u8; 5]);
        std::fs::write(&path, bytes).unwrap();

        let store = SpentBoxes::load(path.clone());
        assert!(!store.spend([1u8; 16]).unwrap(), "read back");
        assert!(!store.spend([2u8; 16]).unwrap(), "read back");
        assert!(store.spend([3u8; 16]).unwrap(), "new, and written in step");
        drop(store);
        released(&path);
        assert!(!SpentBoxes::load(path).spend([3u8; 16]).unwrap());
    }

    #[test]
    fn a_write_the_disk_cut_short_leaves_the_entries_after_it_in_step() {
        // The process lives on, so nothing repaired the file at a start: k2 and k3 were read back
        // out of step, as fresh, and a replay of either opened after the next restart.
        let dir = crate::secret_file::testing::TempDir::new("e2espent-short");
        let path = dir.0.join(SPENT_KEYS_FILE);
        let store = SpentBoxes::load(path.clone());
        assert!(store.spend([1u8; 16]).unwrap());
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[9u8; 7]).unwrap();
        drop(file);
        assert!(store.spend([2u8; 16]).unwrap());
        assert!(store.spend([3u8; 16]).unwrap());
        drop(store);
        released(&path);
        let store = SpentBoxes::load(path);
        for entry in [[1u8; 16], [2u8; 16], [3u8; 16]] {
            assert!(!store.spend(entry).unwrap());
        }
    }

    #[test]
    fn a_second_process_on_the_folder_answers_nothing_until_the_first_has_gone() {
        // A server beside the window, on one data folder: each read the list once, and both opened a
        // recorded row at counter 0.
        let dir = crate::secret_file::testing::TempDir::new("e2espent-two");
        let path = dir.0.join(SPENT_KEYS_FILE);
        let first = SpentBoxes::load(path.clone());
        let second = SpentBoxes::load(path.clone());
        assert!(first.spend([1u8; 16]).unwrap());
        assert!(second.spend([1u8; 16]).unwrap_err().contains("another OAIY process"));
        assert!(second.hold_now().unwrap_err().contains("another OAIY process"), "a lane asks this before it claims");
        drop(first);
        released(&path);
        second.hold_now().unwrap();
        assert!(!second.spend([1u8; 16]).unwrap(), "and it reads what the first spent");
    }

    #[test]
    fn every_spelling_of_a_key_is_one_box() {
        // A public key plus a point of small order makes the same box for any secret that clears
        // that order, about half of all identities, and each is a new string. Run on the real module,
        // a recorded request opened again for 976 identities of 2000.
        use curve25519_dalek::constants::EIGHT_TORSION;
        use curve25519_dalek::montgomery::MontgomeryPoint;
        let mut same_box = 0;
        for seed in 1..=64u8 {
            let desktop = identity(seed);
            let (eph, envelope) = sealed_request(&desktop, 0xB8);
            let sessions = E2eSessions::new();
            sessions.open_inbound(&desktop, "req-1", &eph, &envelope).unwrap();
            let public = decode_b64_fixed::<32>(&eph, "ephPub").unwrap();
            let point = MontgomeryPoint(public).to_edwards(0).unwrap();
            for torsion in EIGHT_TORSION.iter().skip(1) {
                let spelled = (point + torsion).to_montgomery().to_bytes();
                let refused = sessions.open_inbound(&desktop, "req-2", &B64.encode(spelled), &envelope);
                match refused {
                    Ok(_) => panic!("identity {seed}: a recorded request opened a second time"),
                    Err(e) if e.message().contains("used before") => same_box += 1,
                    Err(e) => assert!(e.message().contains("authentication failed"), "{}", e.message()),
                }
            }
        }
        assert!(same_box > 0, "no identity here clears a small order, so this proved nothing");
    }

    #[test]
    fn nothing_is_forgotten_and_at_the_limit_nothing_new_opens() {
        // Rows of junk would otherwise push a recorded request out of the list: the rate is the
        // backend's. It can stop this desktop answering, which it could do by dropping rows.
        let dir = crate::secret_file::testing::TempDir::new("e2espent-limit");
        let mut store = SpentBoxes::load(dir.0.join(SPENT_KEYS_FILE));
        store.limit = 2;
        assert!(store.spend([1u8; 16]).unwrap());
        assert!(store.spend([2u8; 16]).unwrap());
        assert!(store.spend([3u8; 16]).unwrap_err().contains("identity"));
        assert!(!store.spend([1u8; 16]).unwrap(), "what was spent stays spent");
        assert!(!store.spend([2u8; 16]).unwrap());
    }

    #[test]
    fn a_store_that_cannot_be_read_seals_nothing_until_it_can() {
        // A folder in its place: the file exists and cannot be read.
        let dir = crate::secret_file::testing::TempDir::new("e2espent-broken");
        let path = dir.0.join(SPENT_KEYS_FILE);
        std::fs::create_dir_all(&path).unwrap();
        let store = SpentBoxes::load(path.clone());
        assert!(store.spend([1u8; 16]).is_err());
        assert!(store.spend([1u8; 16]).is_err());
        // A scanner lets go, the disk is freed: no restart needed.
        std::fs::remove_dir_all(&path).unwrap();
        assert!(store.spend([1u8; 16]).unwrap());
    }

    #[test]
    fn the_identity_survives_a_restart_and_a_broken_one_is_not_silently_rotated() {
        let dir = std::env::temp_dir().join(format!("oaiy-e2e-{}", uuid::Uuid::new_v4().simple()));
        let first = E2eIdentity::load_or_create(&dir).unwrap();
        let second = E2eIdentity::load_or_create(&dir).unwrap();
        assert_eq!(first.public_key_b64(), second.public_key_b64());

        // Every browser has pinned the public half. Minting a new one because
        // the file went unreadable would show them a key-changed warning they
        // cannot tell from an attack — so it must fail loudly instead.
        std::fs::write(dir.join(IDENTITY_KEY_FILE), b"not a key").unwrap();
        // Matched rather than unwrap_err'd: the identity type holds a secret
        // and deliberately implements no Debug, so it must never be printed.
        let err = match E2eIdentity::load_or_create(&dir) {
            Ok(_) => panic!("a broken identity must fail loudly, not rotate"),
            Err(e) => e,
        };
        assert!(err.contains("trust it again"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_chat_identity_key_is_written_owner_only_and_whole() {
        let dir = crate::secret_file::testing::TempDir::new("e2ekey");
        let identity = E2eIdentity::load_or_create(&dir.0.join("data")).unwrap();
        let key = dir.0.join("data").join(IDENTITY_KEY_FILE);
        crate::secret_file::testing::assert_private(&key);
        crate::secret_file::testing::assert_private_dir(&dir.0.join("data"));
        let stored = E2eIdentity::decode_secret(std::fs::read_to_string(&key).unwrap().trim()).unwrap();
        assert_eq!(stored.public_key_b64(), identity.public_key_b64(), "the file holds this identity's whole secret");
        let names: Vec<_> = std::fs::read_dir(dir.0.join("data")).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names.len(), 1, "no staging file is left beside the key: {names:?}");
    }

    #[test]
    fn the_lanes_of_one_data_folder_share_their_spent_keys() {
        // The tunnel and the sealed flows lane each load the identity themselves.
        let dir = crate::secret_file::testing::TempDir::new("e2espent-lanes");
        let data = dir.0.join("data");
        let tunnel = E2eIdentity::load_or_create(&data).unwrap();
        let flows = E2eIdentity::load_or_create(&data).unwrap();
        assert!(tunnel.spend_request_key(&[7u8; 32]).unwrap());
        assert!(!flows.spend_request_key(&[7u8; 32]).unwrap());
    }

    #[test]
    fn an_oversized_envelope_is_refused_before_it_is_decrypted() {
        let huge = B64.encode(vec![0u8; 24 + MAX_ENVELOPE_CIPHERTEXT_BYTES + 1]);
        assert!(unpack_envelope(&huge).is_err());
        assert!(unpack_envelope(&B64.encode([0u8; 20])).is_err(), "shorter than nonce + tag");
        assert!(unpack_envelope("not base64!!").is_err());
    }
}
