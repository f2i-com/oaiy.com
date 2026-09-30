//! Telling the owner, on the GUI, that a caller is asking for them or that a message was left: a native
//! notification (the notification plugin the downloads already use), and for a ring the window brought up so the
//! ring dialog is in front of them. Nothing here says what the caller said: a notification is read on a lock
//! screen, and the words are for the dialog and the Messages page.
//!
//! The notification plugin has no way to withdraw a notification on a desktop, so a ring that ends leaves its
//! notification to fade by itself; the dialog is what closes. Nor can it say whether the system showed one (it
//! answers that permission is granted whatever Focus Assist or a quiet-hours setting does): what a message's
//! `notified` says is only what the plugin said, that it accepted the notification.

use std::sync::Arc;

use tauri::AppHandle;
use tauri_plugin_notification::NotificationExt;

use crate::messages::{Message, MessageNotifier};
use crate::ring::{ActiveRing, RingNotifier};

/// The longest a caller's name or number is shown in a notification.
const SHOWN: usize = 40;

/// What shows a notification (the notification plugin; a test's).
pub trait Toast: Send + Sync {
    /// Show one. An error says why it could not be shown.
    fn show(&self, title: &str, body: &str) -> Result<(), String>;
}

impl Toast for AppHandle {
    fn show(&self, title: &str, body: &str) -> Result<(), String> {
        self.notification().builder().title(title).body(body).show().map_err(|e| e.to_string())
    }
}

/// What brings the window up, for a ring.
pub type Raise = Arc<dyn Fn() + Send + Sync>;

/// `text` as a notification may show it: control and direction-changing characters gone (a caller's name is theirs to choose, and
/// arrives from the phone), line breaks and runs of spaces one space, and at most [`SHOWN`] characters.
fn shown(text: &str) -> String {
    crate::messages::clean(text, SHOWN)
}

/// Who to say is calling: the name they gave, else the number, else nothing to go by.
fn who(name: &str, number: &str) -> String {
    match (shown(name).as_str(), shown(number).as_str()) {
        ("", "") => "A caller".to_string(),
        ("", number) => number.to_string(),
        (name, "") => name.to_string(),
        (name, number) => format!("{name} ({number})"),
    }
}

/// The notification and window for a ring.
pub struct GuiRing {
    toast: Arc<dyn Toast>,
    raise: Raise,
}

impl GuiRing {
    pub fn new(toast: Arc<dyn Toast>, raise: Raise) -> Self {
        Self { toast, raise }
    }

    /// For the desktop's own window.
    pub fn of(app: AppHandle) -> Self {
        let window = app.clone();
        Self::new(Arc::new(app), Arc::new(move || crate::tray::show_main(&window)))
    }
}

impl RingNotifier for GuiRing {
    fn ringing(&self, ring: &ActiveRing) {
        let body = format!("{} is asking for you. Open OAIY to see who and to answer on your Companion.", who(&ring.caller_name, &ring.caller_number));
        if let Err(e) = self.toast.show("A caller wants to speak to you", &body) {
            log::warn!("ring: the notification could not be shown: {e}");
        }
        (self.raise)();
    }

    fn ended(&self, _id: &str, _outcome: &str) {}

    /// Somebody asked for the owner and no device was set up to take a transfer: told, without bringing the window up
    /// (nothing rings, and there is nothing to answer).
    fn noticed(&self, notice: &crate::ring::Notice) {
        // Why nobody was rung is what the desktop found (never that no device is set up when a phone is approved and only set not to ring).
        let body = format!("{} asked for you. {}", who(&notice.caller_name, &notice.caller_number), notice.text);
        if let Err(e) = self.toast.show("Someone asked for you", &body) {
            log::warn!("ring: the notification could not be shown: {e}");
        }
    }
}

/// The notification for a message.
pub struct GuiMessages(Arc<dyn Toast>);

impl GuiMessages {
    pub fn new(toast: Arc<dyn Toast>) -> Self {
        Self(toast)
    }

    pub fn of(app: AppHandle) -> Self {
        Self::new(Arc::new(app))
    }
}

impl MessageNotifier for GuiMessages {
    /// Whether the notification plugin accepted the notification (which is all it can say).
    fn message_taken(&self, message: &Message) -> bool {
        match self.0.show("A message for you", &format!("{} left a message.", who(&message.name, &message.from))) {
            Ok(()) => true,
            Err(e) => {
                log::warn!("messages: the notification could not be shown: {e}");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Toasts {
        shown: Mutex<Vec<(String, String)>>,
        fail: Mutex<Option<String>>,
    }

    impl Toast for Toasts {
        fn show(&self, title: &str, body: &str) -> Result<(), String> {
            if let Some(why) = self.fail.lock().unwrap().clone() {
                return Err(why);
            }
            self.shown.lock().unwrap().push((title.to_string(), body.to_string()));
            Ok(())
        }
    }

    fn ring(name: &str, number: &str) -> ActiveRing {
        ActiveRing { id: "assist_1".into(), call_id: "call_1".into(), caller_name: name.into(), caller_number: number.into(), said: vec!["Can I speak to the owner?".into()], started_at: 0, expires_at: 30_000, now: 0, devices: vec![], stopping: false, taken: false, note: String::new() }
    }

    fn message(name: &str, from: &str) -> Message {
        serde_json::from_value(serde_json::json!({"id": "msg_1", "at": "2026-09-30T00:00:00Z", "callId": "call_1", "from": from, "name": name, "callback": "", "message": "Ring me.", "urgency": "normal", "wantsCallback": false, "state": "new"})).unwrap()
    }

    #[test]
    fn a_caller_is_named_as_well_as_they_are_known() {
        assert_eq!(who("Alex", "0491 570 006"), "Alex (0491 570 006)");
        assert_eq!(who("  ", "0491 570 006"), "0491 570 006");
        assert_eq!(who("Alex", ""), "Alex");
        assert_eq!(who("", " "), "A caller");
    }

    #[test]
    fn a_name_from_the_phone_is_cleaned_and_cut_before_it_is_shown() {
        // Control characters, line breaks and direction tricks cannot dress a notification up as another.
        assert_eq!(who("Alex\nURGENT: click here\u{7}\u{202e}", ""), "Alex URGENT: click here");
        assert_eq!(who("\u{0}\u{1b}[31m", "\u{200b}"), "[31m", "an escape sequence is text: its escape is dropped");
        let long = who(&"x".repeat(500), &"9".repeat(500));
        assert!(long.chars().count() <= 2 * SHOWN + 3, "{}", long.chars().count());
        assert!(!long.contains('\n'));
        // A name that is only control characters is no name.
        assert_eq!(who("\u{7}\u{8}", "0491 570 006"), "0491 570 006");
    }

    #[test]
    fn a_ring_tells_the_owner_who_and_brings_the_window_up_and_never_what_they_said() {
        let toast = Arc::new(Toasts::default());
        let raised = Arc::new(Mutex::new(0));
        let r = raised.clone();
        let gui = GuiRing::new(toast.clone(), Arc::new(move || *r.lock().unwrap() += 1));
        gui.ringing(&ring("Alex\nInjected: line", "+61491570006"));
        let shown = toast.shown.lock().unwrap().clone();
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].0, "A caller wants to speak to you");
        assert!(shown[0].1.starts_with("Alex Injected: line (+61491570006) is asking for you."), "{}", shown[0].1);
        assert!(!shown[0].1.contains("owner?") && !shown[0].1.contains('\n'), "what they said stays out of a notification: {}", shown[0].1);
        assert_eq!(*raised.lock().unwrap(), 1);
        // A notification that cannot be shown still brings the window up: the dialog is the ring.
        *toast.fail.lock().unwrap() = Some("no notification service".into());
        gui.ringing(&ring("Sam", ""));
        assert_eq!(*raised.lock().unwrap(), 2);
    }

    #[test]
    fn a_notice_of_nobody_to_ring_does_not_bring_the_window_up() {
        let toast = Arc::new(Toasts::default());
        let raised = Arc::new(Mutex::new(0));
        let r = raised.clone();
        let gui = GuiRing::new(toast.clone(), Arc::new(move || *r.lock().unwrap() += 1));
        let notice = |cause: crate::ring::Cause| crate::ring::Notice { id: "notice_1".into(), call_id: "call_1".into(), caller_name: "Alex".into(), caller_number: String::new(), at: 0, text: cause.notice_text().to_string(), cause: cause.code() };
        gui.noticed(&notice(crate::ring::Cause::PhonesOff));
        assert_eq!(toast.shown.lock().unwrap()[0].0, "Someone asked for you");
        assert_eq!(toast.shown.lock().unwrap()[0].1, "Alex asked for you. Your phone is set to not ring, so they were offered a message.");
        // The words say what was found: never that no device is set up when one is, and only set not to ring.
        gui.noticed(&notice(crate::ring::Cause::NoCompanion));
        assert!(toast.shown.lock().unwrap()[1].1.contains("No Companion is approved"));
        assert!(!toast.shown.lock().unwrap()[0].1.contains("No device is set up"));
        assert_eq!(*raised.lock().unwrap(), 0);
    }

    #[test]
    fn a_message_is_notified_only_when_the_plugin_accepted_the_notification() {
        let toast = Arc::new(Toasts::default());
        let messages = GuiMessages::new(toast.clone());
        assert!(messages.message_taken(&message("Alex", "+61491570006")));
        assert_eq!(toast.shown.lock().unwrap()[0].1, "Alex (+61491570006) left a message.");
        // It could not be shown: the receptionist is not told the owner will be told.
        *toast.fail.lock().unwrap() = Some("the notification service is not running".into());
        assert!(!messages.message_taken(&message("Alex", "+61491570006")));
        assert_eq!(toast.shown.lock().unwrap().len(), 1);
    }
}
