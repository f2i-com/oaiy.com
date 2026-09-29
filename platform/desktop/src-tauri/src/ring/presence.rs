//! Whether the owner is at the computer, from how long ago they last used it.
//!
//! On Windows the system says when the last keyboard or mouse input was, which costs nothing to ask,
//! so it is asked when a request to reach the owner is judged (there is no sampler thread to run).
//! Within the owner's `desktopActiveSeconds` (120 unless they say) they are `active`; longer ago,
//! `idle`. Where the system does not say (another OS, the headless server) they are `off`: never a
//! reason to ring this computer. Presence is a hint for the ring policy and nothing else.

use super::plan::Presence;
use super::settings::SettingsStore;

/// How long ago the owner last used the computer, in seconds; none when the system does not say.
pub trait IdleClock: Send + Sync {
    fn idle_seconds(&self) -> Option<u64>;
}

/// The operating system's idea of it.
pub struct OsIdle;

impl IdleClock for OsIdle {
    #[cfg(windows)]
    fn idle_seconds(&self) -> Option<u64> {
        use windows_sys::Win32::System::SystemInformation::GetTickCount;
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetLastInputInfo, LASTINPUTINFO};
        let mut info = LASTINPUTINFO { cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32, dwTime: 0 };
        // SAFETY: `info` is a valid LASTINPUTINFO with its size set, as the call requires.
        let known = unsafe { GetLastInputInfo(&mut info) } != 0;
        // SAFETY: GetTickCount takes no arguments and cannot fail.
        let now = unsafe { GetTickCount() };
        // Both are milliseconds since the system started, and wrap together every 49 days.
        known.then(|| u64::from(now.wrapping_sub(info.dwTime)) / 1000)
    }

    #[cfg(not(windows))]
    fn idle_seconds(&self) -> Option<u64> {
        None
    }
}

/// Presence from an idle clock and the owner's own window.
pub struct IdlePresence {
    idle: Box<dyn IdleClock>,
    settings: SettingsStore,
}

impl IdlePresence {
    pub fn new(idle: Box<dyn IdleClock>, settings: SettingsStore) -> Self {
        Self { idle, settings }
    }

    /// Presence from the operating system.
    pub fn os(settings: SettingsStore) -> Self {
        Self::new(Box::new(OsIdle), settings)
    }
}

impl super::host::PresenceSource for IdlePresence {
    fn presence(&self) -> Presence {
        match self.idle.idle_seconds() {
            None => Presence::Off,
            Some(idle) if idle < u64::from(self.settings.get().desktop_active_seconds) => Presence::Active,
            Some(_) => Presence::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::host::PresenceSource;
    use crate::ring::settings::RingSettings;

    struct Idle(Option<u64>);

    impl IdleClock for Idle {
        fn idle_seconds(&self) -> Option<u64> {
            self.0
        }
    }

    fn presence(idle: Option<u64>, active_seconds: u32) -> Presence {
        let settings = SettingsStore::in_memory(RingSettings { desktop_active_seconds: active_seconds, ..Default::default() });
        IdlePresence::new(Box::new(Idle(idle)), settings).presence()
    }

    #[test]
    fn recent_input_is_active_and_old_input_is_idle_by_the_owners_own_window() {
        assert_eq!(presence(Some(0), 120), Presence::Active);
        assert_eq!(presence(Some(119), 120), Presence::Active);
        assert_eq!(presence(Some(120), 120), Presence::Idle, "the window's end is outside it");
        assert_eq!(presence(Some(100_000), 120), Presence::Idle);
        assert_eq!(presence(Some(200), 300), Presence::Active, "a longer window");
        assert_eq!(presence(Some(31), 30), Presence::Idle, "and the shortest, 30 s");
    }

    #[test]
    fn where_the_system_does_not_say_the_owner_is_never_taken_to_be_here() {
        assert_eq!(presence(None, 120), Presence::Off);
    }

    #[cfg(windows)]
    #[test]
    fn the_system_answers_on_windows() {
        // Whoever runs the tests is at a session: it says a number (a service with no session may not).
        let idle = OsIdle.idle_seconds();
        assert!(idle.is_none_or(|s| s < 60 * 60 * 24 * 60), "{idle:?}");
    }
}
