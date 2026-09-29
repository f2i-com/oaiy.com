//! The updater's state: what the app knows about a newer release, and every move it may make.
//!
//! One [`Updater`] is shared (an `Arc`) by the HTTP routes, the desktop's commands and the tray.
//! It owns the status the window reads, the rate limit on checks, and the verified installer once
//! there is one, and it is the only thing that moves between the states:
//!
//! ```text
//!  idle / upToDate / failed ──check──▶ checking ──▶ upToDate | available | idle (no build for this platform) | failed
//!  available ──download──▶ downloading ──▶ ready | failed
//!  ready ──install (no blockers)──▶ installing ──▶ (the app exits, or) failed
//! ```
//!
//! Nothing here touches the network or the disk beyond the feed read in [`super::check`]; the GUI
//! build adds the download and the hand-off to the installer around it (`update::gui`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;

use super::blockers::{self, Activity, Blocker};
use super::feed::{self, Release, Verdict};
use super::verify::VerifiedPackage;
use super::{check, FeedSource};

/// Checks are at least this far apart, however they are asked for (the window, the tray, the API, the clock).
pub const MIN_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// The releases page: where a person downloads OAIY by hand.
pub const RELEASES_URL: &str = "https://github.com/f2i-com/oaiy.com/releases/latest";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum State {
    Idle,
    Checking,
    UpToDate,
    Available,
    Downloading,
    Ready,
    Installing,
    Failed,
}

/// Which step failed.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Stage {
    Check,
    Download,
    Install,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub downloaded: u64,
    pub total: Option<u64>,
}

/// Whether this build can replace itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoUpdate {
    Yes,
    /// It cannot, and why (shown next to the link to the releases page).
    No(String),
}

pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// What differs between the desktop and the headless server.
pub trait Platform: Send + Sync {
    fn auto_update(&self) -> AutoUpdate;
    /// A newer release for this platform was found: do what the platform needs to install it later
    /// (the desktop asks the updater plugin for its handle on the release). An error is a failed check.
    fn prepare<'a>(&'a self, release: &'a Release) -> BoxFuture<'a, Result<(), String>>;
}

/// The headless server: it tells the owner a newer release exists, and never replaces itself.
pub struct NotifyOnly;

impl Platform for NotifyOnly {
    fn auto_update(&self) -> AutoUpdate {
        AutoUpdate::No("This is the headless server: it tells you when a newer release exists but never replaces itself. Upgrade it by hand (see docs/UPDATES.md).".into())
    }

    fn prepare<'a>(&'a self, _release: &'a Release) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

/// What the window reads (`GET /api/update/status`).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub state: State,
    pub current_version: String,
    /// Always the stable releases.
    pub channel: &'static str,
    pub latest_version: Option<String>,
    pub notes: Option<String>,
    pub published_at: Option<String>,
    pub last_checked_at: Option<String>,
    /// The last thing that went wrong, in words for a person.
    pub error: Option<String>,
    pub failed_during: Option<Stage>,
    pub progress: Option<Progress>,
    /// Something to say beside the state (no release for this platform, say).
    pub note: Option<String>,
    /// This build can download and install an update itself.
    pub can_auto_update: bool,
    /// When it cannot, why: the window shows it beside the link to the releases page.
    pub manual_reason: Option<String>,
    /// What stops an install now (empty: nothing does). Only worked out where an install is possible.
    pub blockers: Vec<Blocker>,
    /// The releases page, for a manual download.
    pub manual_url: &'static str,
    /// Whether the desktop looks for updates by itself (at start and daily).
    pub auto_check: bool,
    /// Seconds until a check is allowed again (None: now).
    pub next_check_in: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckRefusal {
    AlreadyChecking,
    /// Downloading or installing: not now.
    Busy(State),
    /// A check was made less than [`MIN_CHECK_INTERVAL`] ago.
    TooSoon { retry_in: u64 },
}

impl std::fmt::Display for CheckRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckRefusal::AlreadyChecking => write!(f, "A check is already running."),
            CheckRefusal::Busy(State::Downloading) => write!(f, "The update is downloading; checking again would not change it."),
            CheckRefusal::Busy(_) => write!(f, "The update is being installed."),
            CheckRefusal::TooSoon { retry_in } => write!(f, "OAIY checked a moment ago. You can check again in {retry_in} seconds."),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadRefusal {
    NothingToDownload,
    NotPossibleHere(String),
    AlreadyDownloading,
}

impl std::fmt::Display for DownloadRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadRefusal::NothingToDownload => write!(f, "There is no update to download. Check for updates first."),
            DownloadRefusal::NotPossibleHere(why) => write!(f, "{why}"),
            DownloadRefusal::AlreadyDownloading => write!(f, "The update is already downloading."),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallRefusal {
    /// There is no verified update to install.
    NotReady,
    NotPossibleHere(String),
    /// Something is in the way; each reason is in words.
    Blocked(Vec<Blocker>),
}

impl std::fmt::Display for InstallRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallRefusal::NotReady => write!(f, "There is no downloaded update to install. Download it first."),
            InstallRefusal::NotPossibleHere(why) => write!(f, "{why}"),
            InstallRefusal::Blocked(blockers) => write!(f, "OAIY cannot restart now: {}", blockers.iter().map(|b| b.message.as_str()).collect::<Vec<_>>().join(" ")),
        }
    }
}

struct Inner {
    state: State,
    latest: Option<String>,
    notes: Option<String>,
    published_at: Option<String>,
    last_checked_at: Option<DateTime<Utc>>,
    error: Option<String>,
    stage: Option<Stage>,
    progress: Option<Progress>,
    note: Option<String>,
    last_check_started: Option<Instant>,
    /// The release a download would fetch.
    release: Option<Release>,
    /// The verified installer, once downloaded.
    package: Option<VerifiedPackage>,
}

impl Inner {
    fn new() -> Inner {
        Inner { state: State::Idle, latest: None, notes: None, published_at: None, last_checked_at: None, error: None, stage: None, progress: None, note: None, last_check_started: None, release: None, package: None }
    }
}

pub struct Updater {
    current: String,
    started: Instant,
    feed: FeedSource,
    inner: Mutex<Inner>,
    activity: RwLock<Option<Arc<dyn Activity>>>,
    platform: RwLock<Arc<dyn Platform>>,
    /// The feed's key for the platform this runs on (None: no release is built for it).
    platform_key: RwLock<Option<&'static str>>,
    /// The agent page's flush handshake: the nonce it must answer with, and where its answer goes.
    flush: Mutex<Option<(String, mpsc::Sender<()>)>>,
    auto_check: AtomicBool,
}

pub type UpdaterHandle = Arc<Updater>;

/// Ends a check that never reported.
struct CheckGuard<'a> {
    updater: &'a Updater,
    done: bool,
}

impl Drop for CheckGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.updater.finish_check(Err("The check was interrupted.".into()));
        }
    }
}

impl Updater {
    /// An updater for the running version `current`, reading the feed at `feed`.
    pub fn new(current: impl Into<String>, feed: FeedSource, started: Instant) -> UpdaterHandle {
        Arc::new(Updater {
            current: current.into(),
            started,
            feed,
            inner: Mutex::new(Inner::new()),
            activity: RwLock::new(None),
            platform: RwLock::new(Arc::new(NotifyOnly)),
            platform_key: RwLock::new(feed::current_platform_key()),
            flush: Mutex::new(None),
            auto_check: AtomicBool::new(true),
        })
    }

    pub fn current_version(&self) -> &str {
        &self.current
    }

    /// Say what the app is doing (see [`super::blockers`]); until then an install is "still starting".
    pub fn set_activity(&self, activity: Arc<dyn Activity>) {
        *self.activity.write().unwrap_or_else(|e| e.into_inner()) = Some(activity);
    }

    /// Say what this build can do about an update (the desktop replaces the headless default).
    pub fn set_platform(&self, platform: Arc<dyn Platform>) {
        *self.platform.write().unwrap_or_else(|e| e.into_inner()) = platform;
    }

    /// Which platform's entry in the feed is this one's (a test sets it; a build has its own).
    pub fn set_platform_key(&self, key: Option<&'static str>) {
        *self.platform_key.write().unwrap_or_else(|e| e.into_inner()) = key;
    }

    pub fn set_auto_check(&self, on: bool) {
        self.auto_check.store(on, Ordering::SeqCst);
    }

    pub fn auto_check(&self) -> bool {
        self.auto_check.load(Ordering::SeqCst)
    }

    fn platform(&self) -> Arc<dyn Platform> {
        self.platform.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What stops an install now, for the status a window polls: a source may answer from what it read a few seconds ago.
    pub fn blockers(&self, now: Instant) -> Vec<Blocker> {
        self.blockers_with(now, false)
    }

    /// What stops an install now, for the DECISION to install (and the look just before something is stopped):
    /// every source is asked again, none answers from a cache.
    pub fn blockers_fresh(&self, now: Instant) -> Vec<Blocker> {
        self.blockers_with(now, true)
    }

    fn blockers_with(&self, now: Instant, fresh: bool) -> Vec<Blocker> {
        let activity = self.activity.read().unwrap_or_else(|e| e.into_inner()).clone();
        let readings = activity.as_ref().map(|a| a.read(fresh));
        blockers::compute(readings.as_ref(), now.saturating_duration_since(self.started))
    }

    pub fn status(&self) -> Status {
        self.status_at(Instant::now())
    }

    pub fn status_at(&self, now: Instant) -> Status {
        let auto = self.platform().auto_update();
        let can_auto_update = auto == AutoUpdate::Yes;
        // Where an update cannot be installed there is nothing to wait for: no blockers are worked out.
        let blockers = if can_auto_update { self.blockers(now) } else { Vec::new() };
        let inner = self.lock();
        let next_check_in = inner.last_check_started.and_then(|at| {
            let since = now.saturating_duration_since(at);
            (since < MIN_CHECK_INTERVAL).then(|| (MIN_CHECK_INTERVAL - since).as_secs().max(1))
        });
        Status {
            state: inner.state,
            current_version: self.current.clone(),
            channel: "stable",
            latest_version: inner.latest.clone(),
            notes: inner.notes.clone().filter(|n| !n.is_empty()),
            published_at: inner.published_at.clone(),
            last_checked_at: inner.last_checked_at.map(|t| t.to_rfc3339_opts(SecondsFormat::Secs, true)),
            error: inner.error.clone(),
            failed_during: if inner.error.is_some() { inner.stage } else { None },
            progress: inner.progress,
            note: inner.note.clone(),
            can_auto_update,
            manual_reason: match auto {
                AutoUpdate::Yes => None,
                AutoUpdate::No(why) => Some(why),
            },
            blockers,
            manual_url: RELEASES_URL,
            auto_check: self.auto_check(),
            next_check_in,
        }
    }

    // ---- checking ---------------------------------------------------------------------------

    /// Start a check (moves to `checking`), or say why not.
    pub fn begin_check(&self, now: Instant) -> Result<(), CheckRefusal> {
        let mut inner = self.lock();
        match inner.state {
            State::Checking => return Err(CheckRefusal::AlreadyChecking),
            State::Downloading | State::Installing => return Err(CheckRefusal::Busy(inner.state)),
            _ => {}
        }
        if let Some(at) = inner.last_check_started {
            let since = now.saturating_duration_since(at);
            if since < MIN_CHECK_INTERVAL {
                return Err(CheckRefusal::TooSoon { retry_in: (MIN_CHECK_INTERVAL - since).as_secs().max(1) });
            }
        }
        inner.last_check_started = Some(now);
        inner.state = State::Checking;
        Ok(())
    }

    /// A check ended: `outcome` is what the feed said, or why it could not be read.
    pub fn finish_check(&self, outcome: Result<Verdict, String>) {
        let mut inner = self.lock();
        inner.last_checked_at = Some(Utc::now());
        match outcome {
            Ok(Verdict::UpToDate { latest }) => {
                inner.package = None;
                inner.release = None;
                inner.latest = Some(latest);
                inner.notes = None;
                inner.published_at = None;
                inner.note = None;
                inner.error = None;
                inner.stage = None;
                inner.progress = None;
                inner.state = State::UpToDate;
            }
            Ok(Verdict::NoUpdateForPlatform { latest }) => {
                inner.package = None;
                inner.release = None;
                inner.note = Some(format!("A newer version ({latest}) exists, but there is no release of it for this platform. Look on the releases page."));
                inner.latest = Some(latest);
                inner.notes = None;
                inner.published_at = None;
                inner.error = None;
                inner.stage = None;
                inner.progress = None;
                inner.state = State::Idle;
            }
            Ok(Verdict::Available(release)) => {
                // A package already downloaded for this very version stays; another one is dropped.
                let same = inner.package.as_ref().is_some_and(|p| p.version() == release.version);
                if !same {
                    inner.package = None;
                    inner.progress = None;
                }
                inner.latest = Some(release.version.clone());
                inner.notes = Some(release.notes.clone());
                inner.published_at = release.published_at.clone();
                inner.release = Some(release);
                inner.note = None;
                inner.error = None;
                inner.stage = None;
                inner.state = if same { State::Ready } else { State::Available };
            }
            Err(message) => {
                inner.error = Some(message);
                inner.stage = Some(Stage::Check);
                // A release already found (and a download already made) is still there to use.
                inner.state = match inner.state {
                    State::Checking if inner.package.is_some() => State::Ready,
                    State::Checking if inner.release.is_some() => State::Available,
                    State::Checking => State::Failed,
                    other => other,
                };
            }
        }
    }

    /// Check the feed: [`begin_check`](Self::begin_check), read it, and record what it says (read it with [`status`](Self::status)).
    pub async fn check(&self) -> Result<(), CheckRefusal> {
        self.begin_check(Instant::now())?;
        // A check whose future is dropped half way (the app is closing a task) must not leave the state on "checking".
        let mut guard = CheckGuard { updater: self, done: false };
        let outcome = self.read_feed().await;
        guard.done = true;
        self.finish_check(outcome);
        Ok(())
    }

    async fn read_feed(&self) -> Result<Verdict, String> {
        let feed = check::fetch(&self.feed).await.map_err(|e| e.to_string())?;
        let platform = self.platform();
        // A build that cannot install (an MSI, a .deb, a headless server) still tells there is something newer.
        let key = *self.platform_key.read().unwrap_or_else(|e| e.into_inner());
        let verdict = feed::evaluate(&feed, &self.current, key);
        if let (Verdict::Available(release), AutoUpdate::Yes) = (&verdict, platform.auto_update()) {
            platform.prepare(release).await?;
        }
        Ok(verdict)
    }

    // ---- downloading ------------------------------------------------------------------------

    /// Start the download of the release found (moves to `downloading`), or say why not.
    pub fn begin_download(&self) -> Result<Release, DownloadRefusal> {
        if let AutoUpdate::No(why) = self.platform().auto_update() {
            return Err(DownloadRefusal::NotPossibleHere(why));
        }
        let mut inner = self.lock();
        match (inner.state, inner.stage) {
            (State::Downloading, _) => return Err(DownloadRefusal::AlreadyDownloading),
            (State::Available, _) | (State::Failed, Some(Stage::Download | Stage::Install)) => {}
            _ => return Err(DownloadRefusal::NothingToDownload),
        }
        let release = inner.release.clone().ok_or(DownloadRefusal::NothingToDownload)?;
        inner.state = State::Downloading;
        inner.progress = Some(Progress { downloaded: 0, total: None });
        inner.error = None;
        inner.stage = None;
        inner.package = None;
        Ok(release)
    }

    pub fn set_progress(&self, downloaded: u64, total: Option<u64>) {
        let mut inner = self.lock();
        if inner.state == State::Downloading {
            inner.progress = Some(Progress { downloaded, total });
        }
    }

    /// The download ended: a verified package (`ready`), or why not (`failed`, and nothing is kept).
    pub fn finish_download(&self, outcome: Result<VerifiedPackage, String>) {
        let mut inner = self.lock();
        if inner.state != State::Downloading {
            return;
        }
        match outcome {
            Ok(package) => {
                inner.package = Some(package);
                inner.progress = None;
                inner.state = State::Ready;
            }
            Err(message) => {
                inner.package = None;
                inner.progress = None;
                inner.error = Some(message);
                inner.stage = Some(Stage::Download);
                inner.state = State::Failed;
            }
        }
    }

    // ---- installing -------------------------------------------------------------------------

    /// Take the verified package to install (moves to `installing`), or say why not. Blockers are looked at here.
    pub fn begin_install(&self, now: Instant) -> Result<VerifiedPackage, InstallRefusal> {
        if let AutoUpdate::No(why) = self.platform().auto_update() {
            return Err(InstallRefusal::NotPossibleHere(why));
        }
        let blockers = self.blockers_fresh(now);
        let mut inner = self.lock();
        if inner.state != State::Ready || inner.package.is_none() {
            return Err(InstallRefusal::NotReady);
        }
        if !blockers.is_empty() {
            return Err(InstallRefusal::Blocked(blockers));
        }
        inner.state = State::Installing;
        inner.error = None;
        inner.stage = None;
        inner.package.take().ok_or(InstallRefusal::NotReady)
    }

    /// The install did not happen (what was stopped has been started again): `failed`, and the download is gone.
    pub fn fail_install(&self, message: String) {
        let mut inner = self.lock();
        inner.package = None;
        inner.error = Some(message);
        inner.stage = Some(Stage::Install);
        inner.state = State::Failed;
    }

    /// The blockers appeared between the button and the stop: back to `ready`, with the package, nothing done.
    pub fn return_to_ready(&self, package: VerifiedPackage) {
        let mut inner = self.lock();
        inner.package = Some(package);
        inner.state = State::Ready;
    }

    // ---- the agent page's flush -------------------------------------------------------------

    /// Arm the agent's flush handshake: the nonce the page has to answer with, and where the answer arrives.
    pub fn arm_flush(&self) -> (String, mpsc::Receiver<()>) {
        let mut bytes = [0u8; 16];
        let _ = getrandom::getrandom(&mut bytes);
        let nonce: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let (tx, rx) = mpsc::channel();
        *self.flush.lock().unwrap_or_else(|e| e.into_inner()) = Some((nonce.clone(), tx));
        (nonce, rx)
    }

    /// The page says it has flushed (`nonce` is the one it was given). False when nothing waits for that nonce.
    pub fn flush_ack(&self, nonce: &str) -> bool {
        let mut waiting = self.flush.lock().unwrap_or_else(|e| e.into_inner());
        match waiting.as_ref() {
            Some((want, _)) if !want.is_empty() && want == nonce => {
                let (_, tx) = waiting.take().expect("just matched");
                let _ = tx.send(());
                true
            }
            _ => false,
        }
    }

    pub fn disarm_flush(&self) {
        *self.flush.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::blockers::fake::Fake;
    use super::super::verify::testing::{comment, Keys};
    use super::*;

    pub const LONG: Duration = Duration::from_secs(3600);

    pub fn release(version: &str) -> Release {
        Release { version: version.into(), notes: format!("Notes for {version}."), published_at: Some("2026-10-01T02:03:04Z".into()), url: format!("https://github.com/f2i-com/oaiy.com/releases/download/v{version}/oaiy-setup.exe"), signature: "sig".into() }
    }

    pub fn package(version: &str) -> VerifiedPackage {
        let keys = Keys::new(7);
        let bytes = b"installer".to_vec();
        let sig = keys.sign(&bytes, &comment(version));
        super::super::verify::verify_package(bytes, &sig, &keys.pubkey, version, true).unwrap()
    }

    struct Auto;
    impl Platform for Auto {
        fn auto_update(&self) -> AutoUpdate {
            AutoUpdate::Yes
        }
        fn prepare<'a>(&'a self, _release: &'a Release) -> BoxFuture<'a, Result<(), String>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// An updater at 0.1.0 that can install and is doing nothing, and "now": an hour after it started.
    pub fn updater() -> (UpdaterHandle, Arc<Fake>, Instant) {
        let started = Instant::now();
        let u = Updater::new("0.1.0", FeedSource::production(), started);
        let fake = Arc::new(Fake::default());
        u.set_activity(fake.clone());
        u.set_platform(Arc::new(Auto));
        (u, fake, started + LONG)
    }

    /// An updater that has found `0.2.0`.
    pub fn available() -> (UpdaterHandle, Arc<Fake>, Instant) {
        let (u, fake, now) = updater();
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        (u, fake, now)
    }

    /// An updater with `0.2.0` downloaded and verified.
    pub fn ready() -> (UpdaterHandle, Arc<Fake>, Instant) {
        let (u, fake, now) = available();
        u.begin_download().unwrap();
        u.finish_download(Ok(package("0.2.0")));
        (u, fake, now)
    }

    #[test]
    fn a_new_updater_is_idle_with_the_running_version_and_the_stable_channel() {
        let (u, _, now) = updater();
        let s = u.status_at(now);
        assert_eq!(s.state, State::Idle);
        assert_eq!(s.current_version, "0.1.0");
        assert_eq!(s.channel, "stable");
        assert_eq!((s.latest_version, s.error, s.last_checked_at, s.progress, s.next_check_in), (None, None, None, None, None));
        assert_eq!(s.manual_url, "https://github.com/f2i-com/oaiy.com/releases/latest");
        assert!(s.can_auto_update && s.blockers.is_empty());
    }

    #[test]
    fn a_check_goes_idle_checking_then_to_what_the_feed_said() {
        let (u, _, now) = updater();
        u.begin_check(now).unwrap();
        assert_eq!(u.status_at(now).state, State::Checking);
        u.finish_check(Ok(Verdict::UpToDate { latest: "0.1.0".into() }));
        let s = u.status_at(now);
        assert_eq!(s.state, State::UpToDate);
        assert_eq!(s.latest_version.as_deref(), Some("0.1.0"));
        assert!(s.last_checked_at.is_some());

        u.begin_check(now + MIN_CHECK_INTERVAL).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        let s = u.status_at(now);
        assert_eq!(s.state, State::Available);
        assert_eq!((s.latest_version.as_deref(), s.notes.as_deref(), s.published_at.as_deref()), (Some("0.2.0"), Some("Notes for 0.2.0."), Some("2026-10-01T02:03:04Z")));
    }

    #[test]
    fn a_platform_with_no_release_is_idle_with_a_note_and_not_an_error() {
        let (u, _, now) = updater();
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::NoUpdateForPlatform { latest: "0.2.0".into() }));
        let s = u.status_at(now);
        assert_eq!(s.state, State::Idle);
        assert!(s.error.is_none());
        assert!(s.note.unwrap().contains("no release of it for this platform"));
        assert_eq!(s.latest_version.as_deref(), Some("0.2.0"));
    }

    #[test]
    fn a_check_that_fails_is_failed_with_the_reason_in_words_and_checking_again_is_allowed_later() {
        let (u, _, now) = updater();
        u.begin_check(now).unwrap();
        u.finish_check(Err("Could not reach the update server.".into()));
        let s = u.status_at(now);
        assert_eq!((s.state, s.failed_during), (State::Failed, Some(Stage::Check)));
        assert_eq!(s.error.as_deref(), Some("Could not reach the update server."));
        assert!(u.begin_check(now + MIN_CHECK_INTERVAL).is_ok());
        u.finish_check(Ok(Verdict::UpToDate { latest: "0.1.0".into() }));
        let s = u.status_at(now);
        assert_eq!((s.state, s.error, s.failed_during), (State::UpToDate, None, None));
    }

    #[test]
    fn a_failed_check_leaves_a_release_already_found_in_place() {
        let (u, _, now) = available();
        u.begin_check(now + MIN_CHECK_INTERVAL).unwrap();
        u.finish_check(Err("offline".into()));
        let s = u.status_at(now);
        assert_eq!(s.state, State::Available);
        assert_eq!(s.error.as_deref(), Some("offline"));
        assert_eq!(s.latest_version.as_deref(), Some("0.2.0"));
        // ...and a downloaded one stays ready.
        let (u, _, now) = ready();
        u.begin_check(now + MIN_CHECK_INTERVAL).unwrap();
        u.finish_check(Err("offline".into()));
        assert_eq!(u.status_at(now).state, State::Ready);
        assert!(u.begin_install(now).is_ok());
    }

    #[test]
    fn checks_are_at_least_thirty_seconds_apart_however_they_are_asked_for() {
        let (u, _, now) = updater();
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::UpToDate { latest: "0.1.0".into() }));
        assert_eq!(u.begin_check(now + Duration::from_secs(1)), Err(CheckRefusal::TooSoon { retry_in: 29 }));
        assert_eq!(u.begin_check(now + Duration::from_secs(29)), Err(CheckRefusal::TooSoon { retry_in: 1 }));
        assert_eq!(u.status_at(now + Duration::from_secs(10)).next_check_in, Some(20));
        assert!(u.begin_check(now + Duration::from_secs(30)).is_ok());
        assert_eq!(u.status_at(now + Duration::from_secs(30)).state, State::Checking);
    }

    #[test]
    fn a_check_is_refused_while_one_runs_and_while_downloading_or_installing() {
        let (u, _, now) = updater();
        u.begin_check(now).unwrap();
        assert_eq!(u.begin_check(now + LONG), Err(CheckRefusal::AlreadyChecking));
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        u.begin_download().unwrap();
        assert_eq!(u.begin_check(now + LONG), Err(CheckRefusal::Busy(State::Downloading)));
        u.finish_download(Ok(package("0.2.0")));
        u.begin_install(now).unwrap();
        assert_eq!(u.begin_check(now + LONG), Err(CheckRefusal::Busy(State::Installing)));
    }

    #[test]
    fn a_download_goes_available_downloading_ready_with_its_progress_on_the_way() {
        let (u, _, now) = available();
        let release = u.begin_download().unwrap();
        assert_eq!(release.version, "0.2.0");
        let s = u.status_at(now);
        assert_eq!((s.state, s.progress), (State::Downloading, Some(Progress { downloaded: 0, total: None })));
        u.set_progress(1_000, Some(4_000));
        assert_eq!(u.status_at(now).progress, Some(Progress { downloaded: 1_000, total: Some(4_000) }));
        u.finish_download(Ok(package("0.2.0")));
        let s = u.status_at(now);
        assert_eq!((s.state, s.progress, s.error), (State::Ready, None, None));
    }

    #[test]
    fn progress_out_of_turn_changes_nothing() {
        let (u, _, now) = available();
        u.set_progress(5, Some(10));
        assert_eq!(u.status_at(now).progress, None);
    }

    #[test]
    fn a_download_that_fails_is_failed_and_keeps_no_package_and_can_be_tried_again() {
        let (u, _, now) = available();
        u.begin_download().unwrap();
        u.finish_download(Err("The downloaded update does not match its signature, so it was thrown away.".into()));
        let s = u.status_at(now);
        assert_eq!((s.state, s.failed_during), (State::Failed, Some(Stage::Download)));
        assert!(s.error.unwrap().contains("thrown away"));
        // It is never installable.
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
        // Trying again is allowed.
        assert!(u.begin_download().is_ok());
        assert_eq!(u.status_at(now).state, State::Downloading);
    }

    #[test]
    fn a_download_is_refused_with_nothing_found_and_while_one_runs() {
        let (u, _, _) = updater();
        assert_eq!(u.begin_download().unwrap_err(), DownloadRefusal::NothingToDownload);
        let (u, _, _) = available();
        u.begin_download().unwrap();
        assert_eq!(u.begin_download().unwrap_err(), DownloadRefusal::AlreadyDownloading);
        let (u, _, _) = ready();
        assert_eq!(u.begin_download().unwrap_err(), DownloadRefusal::NothingToDownload);
    }

    #[test]
    fn an_install_needs_a_verified_download_and_nothing_in_the_way() {
        let (u, fake, now) = available();
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
        u.begin_download().unwrap();
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
        u.finish_download(Ok(package("0.2.0")));
        fake.set(|s| s.calls = 1);
        match u.begin_install(now) {
            Err(InstallRefusal::Blocked(blockers)) => assert_eq!(blockers[0].code, "call"),
            other => panic!("{other:?}"),
        }
        // Nothing changed: still ready, package kept.
        assert_eq!(u.status_at(now).state, State::Ready);
        fake.set(|s| s.calls = 0);
        let package = u.begin_install(now).unwrap();
        assert_eq!(package.version(), "0.2.0");
        assert_eq!(u.status_at(now).state, State::Installing);
        // The package is taken: there is only one install.
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
    }

    #[test]
    fn the_status_may_answer_from_a_cache_but_the_decision_to_install_asks_every_source_again() {
        let (u, fake, now) = ready();
        let before = (fake.reads.load(std::sync::atomic::Ordering::SeqCst), fake.fresh_reads.load(std::sync::atomic::Ordering::SeqCst));
        u.status_at(now);
        u.blockers(now);
        assert_eq!(fake.fresh_reads.load(std::sync::atomic::Ordering::SeqCst), before.1, "a polled status is not a fresh read");
        assert!(fake.reads.load(std::sync::atomic::Ordering::SeqCst) > before.0);
        u.begin_install(now).unwrap();
        assert_eq!(fake.fresh_reads.load(std::sync::atomic::Ordering::SeqCst), before.1 + 1, "the install decision is");
    }

    #[test]
    fn an_install_right_after_start_is_blocked_until_two_minutes_have_passed() {
        let now = Instant::now();
        let u = Updater::new("0.1.0", FeedSource::production(), now);
        u.set_activity(Arc::new(Fake::default()));
        u.set_platform(Arc::new(Auto));
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        u.begin_download().unwrap();
        u.finish_download(Ok(package("0.2.0")));
        assert!(matches!(u.begin_install(now + Duration::from_secs(60)), Err(InstallRefusal::Blocked(b)) if b[0].code == "starting"));
        assert!(u.begin_install(now + Duration::from_secs(121)).is_ok());
    }

    #[test]
    fn an_install_with_nothing_known_of_what_the_app_is_doing_is_blocked() {
        let started = Instant::now();
        let now = started + LONG;
        let u = Updater::new("0.1.0", FeedSource::production(), started);
        u.set_platform(Arc::new(Auto));
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        u.begin_download().unwrap();
        u.finish_download(Ok(package("0.2.0")));
        assert!(matches!(u.begin_install(now), Err(InstallRefusal::Blocked(_))));
    }

    #[test]
    fn a_failed_install_is_failed_and_the_download_is_gone() {
        let (u, _, now) = ready();
        u.begin_install(now).unwrap();
        u.fail_install("The installer could not be started.".into());
        let s = u.status_at(now);
        assert_eq!((s.state, s.failed_during), (State::Failed, Some(Stage::Install)));
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
        // Downloading again is allowed (the release found is still known).
        assert!(u.begin_download().is_ok());
    }

    #[test]
    fn blockers_arriving_between_the_button_and_the_stop_put_it_back_to_ready() {
        let (u, _, now) = ready();
        let package = u.begin_install(now).unwrap();
        u.return_to_ready(package);
        assert_eq!(u.status_at(now).state, State::Ready);
        assert!(u.begin_install(now).is_ok());
    }

    #[test]
    fn a_later_check_that_finds_the_same_version_keeps_the_download_and_a_newer_one_drops_it() {
        let (u, _, now) = ready();
        u.begin_check(now + MIN_CHECK_INTERVAL).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        assert_eq!(u.status_at(now).state, State::Ready);
        u.begin_check(now + MIN_CHECK_INTERVAL * 2).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.3.0"))));
        let s = u.status_at(now);
        assert_eq!((s.state, s.latest_version.as_deref()), (State::Available, Some("0.3.0")));
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
        // And a feed that no longer offers anything drops it too.
        let (u, _, now) = ready();
        u.begin_check(now + MIN_CHECK_INTERVAL).unwrap();
        u.finish_check(Ok(Verdict::UpToDate { latest: "0.1.0".into() }));
        assert_eq!(u.begin_install(now), Err(InstallRefusal::NotReady));
    }

    #[test]
    fn a_platform_that_cannot_install_shows_why_offers_no_download_and_works_out_no_blockers() {
        let now = Instant::now();
        let u = Updater::new("0.1.0", FeedSource::production(), now);
        // NotifyOnly is the default platform (the headless server).
        u.begin_check(now).unwrap();
        u.finish_check(Ok(Verdict::Available(release("0.2.0"))));
        let s = u.status_at(now);
        assert_eq!(s.state, State::Available);
        assert!(!s.can_auto_update && s.blockers.is_empty());
        assert!(s.manual_reason.unwrap().contains("never replaces itself"));
        assert!(matches!(u.begin_download(), Err(DownloadRefusal::NotPossibleHere(_))));
        assert!(matches!(u.begin_install(now + LONG), Err(InstallRefusal::NotPossibleHere(_))));
    }

    #[tokio::test]
    async fn a_build_that_cannot_install_is_not_asked_to_get_ready_for_one_and_one_that_can_is() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting(AutoUpdate, Arc<AtomicUsize>, Option<String>);
        impl Platform for Counting {
            fn auto_update(&self) -> AutoUpdate {
                self.0.clone()
            }
            fn prepare<'a>(&'a self, _release: &'a Release) -> BoxFuture<'a, Result<(), String>> {
                self.1.fetch_add(1, Ordering::SeqCst);
                let failure = self.2.clone();
                Box::pin(async move { failure.map_or(Ok(()), Err) })
            }
        }
        // A one-request server standing in for the release feed.
        let feed = serde_json::json!({"version": "0.2.0", "platforms": {"windows-x86_64": {"signature": "c2ln", "url": "https://github.com/f2i-com/oaiy.com/releases/download/v0.2.0/x.exe"}}}).to_string();
        let app = axum::Router::new().route("/latest.json", axum::routing::get(move || { let feed = feed.clone(); async move { feed } }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/latest.json", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let source = FeedSource { url, insecure: true };
        for (auto, prepared, failure, state) in [
            (AutoUpdate::No("manual".into()), 0, None, State::Available),
            (AutoUpdate::Yes, 1, None, State::Available),
            // Not being able to get ready (the plugin found nothing, say) is a failed check, not an available update.
            (AutoUpdate::Yes, 1, Some("The update information changed. Check again.".to_string()), State::Failed),
        ] {
            let count = Arc::new(AtomicUsize::new(0));
            let u = Updater::new("0.1.0", source.clone(), Instant::now());
            u.set_platform_key(Some("windows-x86_64"));
            u.set_platform(Arc::new(Counting(auto, count.clone(), failure)));
            u.check().await.unwrap();
            assert_eq!((count.load(Ordering::SeqCst), u.status().state), (prepared, state));
        }
    }

    #[test]
    fn the_status_serialises_as_camel_case_names_the_window_reads() {
        let (u, _, now) = available();
        let json = serde_json::to_value(u.status_at(now)).unwrap();
        for key in ["state", "currentVersion", "channel", "latestVersion", "notes", "publishedAt", "lastCheckedAt", "canAutoUpdate", "blockers", "manualUrl", "autoCheck"] {
            assert!(json.get(key).is_some(), "{key} in {json}");
        }
        assert_eq!(json["state"], "available");
        for (state, name) in [(State::UpToDate, "upToDate"), (State::Idle, "idle"), (State::Checking, "checking"), (State::Downloading, "downloading"), (State::Ready, "ready"), (State::Installing, "installing"), (State::Failed, "failed")] {
            assert_eq!(serde_json::to_value(state).unwrap(), name);
        }
    }

    #[test]
    fn the_flush_answer_only_counts_with_the_nonce_it_was_asked_with() {
        let (u, _, _) = updater();
        assert!(!u.flush_ack("anything"), "nothing waits");
        let (nonce, rx) = u.arm_flush();
        assert_eq!(nonce.len(), 32);
        assert!(!u.flush_ack("wrong"));
        assert!(!u.flush_ack(""));
        assert!(rx.try_recv().is_err());
        assert!(u.flush_ack(&nonce));
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
        // Once only.
        assert!(!u.flush_ack(&nonce));
    }
}
