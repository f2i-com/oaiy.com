//! Telling the owner, on the GUI, that a caller is asking for them or that a message was left: a native
//! notification (the notification plugin the downloads already use), and for a ring the window brought up so the
//! ring dialog is in front of them. Nothing here says what the caller said: a notification is read on a lock
//! screen, and the words are for the dialog and the Messages page.
//!
//! The notification plugin has no way to withdraw a notification on a desktop, so a ring that ends leaves its
//! notification to fade by itself; the dialog is what closes.

use tauri::AppHandle;
use tauri_plugin_notification::NotificationExt;

use crate::messages::{Message, MessageNotifier};
use crate::ring::{ActiveRing, RingNotifier};

/// Who to say is calling: the name they gave, else the number, else nothing to go by.
fn who(name: &str, number: &str) -> String {
    match (name.trim(), number.trim()) {
        ("", "") => "A caller".to_string(),
        ("", number) => number.to_string(),
        (name, "") => name.to_string(),
        (name, number) => format!("{name} ({number})"),
    }
}

/// The notification and window for a ring.
pub struct GuiRing(pub AppHandle);

impl RingNotifier for GuiRing {
    fn ringing(&self, ring: &ActiveRing) {
        let shown = self.0.notification().builder().title("A caller wants to speak to you").body(format!("{} is asking for you. Open OAIY to take the call.", who(&ring.caller_name, &ring.caller_number))).show();
        if let Err(e) = shown {
            log::warn!("ring: the notification could not be shown: {e}");
        }
        crate::tray::show_main(&self.0);
    }

    fn ended(&self, _id: &str, _outcome: &str) {}
}

/// The notification for a message.
pub struct GuiMessages(pub AppHandle);

impl MessageNotifier for GuiMessages {
    fn message_taken(&self, message: &Message) -> bool {
        self.0.notification().builder().title("A message for you").body(format!("{} left a message.", who(&message.name, &message.from))).show().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::who;

    #[test]
    fn a_caller_is_named_as_well_as_they_are_known() {
        assert_eq!(who("Alex", "0491 570 006"), "Alex (0491 570 006)");
        assert_eq!(who("  ", "0491 570 006"), "0491 570 006");
        assert_eq!(who("Alex", ""), "Alex");
        assert_eq!(who("", " "), "A caller");
    }
}
