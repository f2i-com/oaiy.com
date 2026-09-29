//! Whether the owner is at the computer, from how long ago they last used it, and whether the screen is locked.
//!
//! On Windows the system says when the last keyboard or mouse input was, which costs nothing to ask,
//! so it is asked when a request to reach the owner is judged (there is no sampler thread to run).
//! A locked workstation is `locked` at once, whatever the input said a moment ago: the owner who pressed
//! Win+L is away, and would otherwise be taken for active for the two minutes after (the secure desktop
//! is the input desktop while it is locked, and a process cannot open that one). Within the owner's
//! `desktopActiveSeconds` (120 unless they say) they are `active`; longer ago, `idle`. Where the system
//! does not say (another OS, the headless server) they are `off`: never a reason to ring this computer.
//! Presence is a hint for the ring policy and nothing else.

use super::plan::Presence;
use super::settings::SettingsStore;

/// How long ago the owner last used the computer, in seconds; none when the system does not say.
pub trait IdleClock: Send + Sync {
    fn idle_seconds(&self) -> Option<u64>;

    /// The workstation is locked (its secure desktop is the one being shown). False where the system does not say.
    fn locked(&self) -> bool {
        false
    }
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

    #[cfg(windows)]
    fn locked(&self) -> bool {
        use windows_sys::Win32::System::StationsAndDesktops::{CloseDesktop, OpenInputDesktop, DESKTOP_SWITCHDESKTOP};
        // While the workstation is locked the input desktop is Winlogon's secure desktop, which a user's process may not open
        // for switching: the call fails. While it is unlocked it succeeds and the handle is closed at once.
        // SAFETY: OpenInputDesktop takes no pointers; a handle it returns is closed here and not used.
        let desktop = unsafe { OpenInputDesktop(0, 0, DESKTOP_SWITCHDESKTOP) };
        if desktop.is_null() {
            return true;
        }
        // SAFETY: `desktop` is a valid handle from the call above, closed once.
        unsafe { CloseDesktop(desktop) };
        false
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
            // Whatever the input said a moment ago, a locked screen is nobody at the computer.
            Some(_) if self.idle.locked() => Presence::Locked,
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

    struct Idle(Option<u64>, bool);

    impl IdleClock for Idle {
        fn idle_seconds(&self) -> Option<u64> {
            self.0
        }
        fn locked(&self) -> bool {
            self.1
        }
    }

    fn presence(idle: Option<u64>, active_seconds: u32) -> Presence {
        locked_presence(idle, false, active_seconds)
    }

    fn locked_presence(idle: Option<u64>, locked: bool, active_seconds: u32) -> Presence {
        let settings = SettingsStore::in_memory(RingSettings { desktop_active_seconds: active_seconds, ..Default::default() });
        IdlePresence::new(Box::new(Idle(idle, locked)), settings).presence()
    }

    #[test]
    fn a_locked_screen_is_nobody_at_the_computer_whatever_the_last_input_was() {
        // Win+L a second after the last keystroke: input says active, the lock says otherwise.
        assert_eq!(locked_presence(Some(1), true, 120), Presence::Locked);
        assert_eq!(locked_presence(Some(500), true, 120), Presence::Locked);
        assert_eq!(locked_presence(Some(1), false, 120), Presence::Active, "unlocked again");
        // Where the system does not say anything, the owner is not taken to be here either way.
        assert_eq!(locked_presence(None, true, 120), Presence::Off);
    }

    #[test]
    fn a_locked_owner_is_away_for_the_ring_policy_and_not_rung_on_this_computer() {
        use crate::ring::plan::{Availability, Device, DeviceKind, Role};
        use crate::ring::{CallInfo, Reason, Ring};
        struct Locked;
        impl PresenceSource for Locked {
            fn presence(&self) -> Presence {
                Presence::Locked
            }
        }
        struct Devices;
        impl crate::ring::DeviceSource for Devices {
            fn devices(&self, _: &RingSettings) -> Vec<Device> {
                vec![
                    Device { id: "pc1".into(), role: Role::SecondDevice, kind: DeviceKind::Windows, call_authority: true, can_take: true, online: true, pushable: false, availability: Availability::Available },
                    Device { id: "ph1".into(), role: Role::SecondDevice, kind: DeviceKind::Android, call_authority: true, can_take: true, online: true, pushable: false, availability: Availability::Available },
                ]
            }
        }
        let ring = Ring::in_memory(RingSettings { enabled: true, ..Default::default() });
        ring.set_presence(std::sync::Arc::new(Locked));
        ring.set_devices(std::sync::Arc::new(Devices));
        let plan = ring.plan_for_plugin("call_1", Reason::CallerAsked, CallInfo { from: "+61491570006".into(), name: String::new(), turns: vec!["Can I speak to the owner".into()] });
        // Away: the phone rings (when_away), and the computer, which nobody is at, does not.
        assert!(plan.rings(), "{:?}", plan.plan);
        assert_eq!((plan.plan.phones.clone(), plan.plan.desktop_toast, plan.plan.desktop_companions.clone()), (vec!["ph1".to_string()], false, Vec::<String>::new()));
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
        // Whoever runs the tests is at an unlocked session (a locked one could not run them interactively; a service or a
        // remote session may say either), so all that is checked is that asking works and does not leak a handle.
        for _ in 0..200 {
            let _ = OsIdle.locked();
        }
    }
}
