use chrono::{Duration, TimeZone, Utc};

use super::*;
use crate::secret_file::testing::{assert_private, TempDir};

fn new(call: &str, from: &str, words: &str) -> NewMessage {
    NewMessage { call_id: call.into(), from: from.into(), name: "Alex".into(), callback: String::new(), message: words.into(), urgent: false, wants_callback: true }
}

fn noon() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 30, 12, 0, 0).unwrap()
}

#[test]
fn words_are_cleaned_and_cut() {
    assert_eq!(clean("  Please\tring\r\nme  about   Friday.  ", 600), "Please ring me about Friday.");
    assert_eq!(clean("a\u{0}b\u{7}c\u{202e}d\u{200b}e\u{feff}f", 600), "abcdef", "control and direction-changing characters are removed, not turned into spaces");
    assert_eq!(clean("\u{1b}[31mred\u{1b}[0m", 600), "[31mred[0m", "an escape sequence is text, not a command: its escape is dropped");
    assert_eq!(clean(&"ab".repeat(400), 600).chars().count(), 600);
    assert_eq!(clean("caf\u{e9} \u{1f600} ok", 600), "caf\u{e9} \u{1f600} ok", "letters and emoji stay");
    assert_eq!(clean("   \n\t ", 600), "");
    assert_eq!(clean_number(" +61 491-570 (006) "), "+61491570006");
    assert_eq!(clean_number("0491 570 006"), "0491570006");
    assert_eq!(clean_number("call me maybe"), "");
    assert_eq!(clean_number("+1234567890123456789012345").len(), MAX_CALLBACK);
    assert_eq!(clean_number("<script>1</script>"), "1");
}

#[test]
fn a_message_is_kept_with_the_number_the_call_came_from() {
    let store = Store::in_memory();
    let m = store.add_at(NewMessage { callback: "0491 570 156".into(), name: "  Alex\n Smith ".into(), urgent: true, ..new("call_1", "+61491570006", "Please ring me about Friday.") }, noon()).unwrap();
    assert!(m.id.starts_with("msg_") && m.id.len() == 20, "{}", m.id);
    assert_eq!((m.at.as_str(), m.call_id.as_str(), m.from.as_str()), ("2026-09-30T12:00:00Z", "call_1", "+61491570006"));
    assert_eq!((m.name.as_str(), m.callback.as_str(), m.message.as_str()), ("Alex Smith", "0491570156", "Please ring me about Friday."));
    assert_eq!((m.urgency, m.wants_callback, m.state), (Urgency::Urgent, true, State::New));
    assert!(m.seen_at.is_none() && m.handled_at.is_none() && m.handled_by.is_none());
    // No callback given: their own number is the one to ring.
    let n = store.add_at(new("call_2", "+61491570006", "Another."), noon()).unwrap();
    assert_eq!((n.callback.as_str(), n.urgency), ("+61491570006", Urgency::Normal));
    // A hidden number: nothing to ring unless they gave one.
    let h = store.add_at(new("call_3", "", "Hidden."), noon()).unwrap();
    assert_eq!((h.from.as_str(), h.callback.as_str()), ("", ""));
    let json = serde_json::to_value(&m).unwrap();
    for key in ["id", "at", "callId", "from", "name", "callback", "message", "urgency", "wantsCallback", "state", "seenAt", "handledAt", "handledBy"] {
        assert!(json.get(key).is_some(), "{key}");
    }
}

#[test]
fn an_empty_message_is_refused_and_the_same_words_on_a_call_are_kept_once() {
    let store = Store::in_memory();
    assert_eq!(store.add(new("call_1", "+61491570006", "  \n ")).unwrap_err().code, "empty_message");
    assert_eq!(store.add(new("call_1", "+61491570006", "\u{200b}\u{202e}")).unwrap_err().code, "empty_message");
    let first = store.add(new("call_1", "+61491570006", "Ring me.")).unwrap();
    let again = store.add(new("call_1", "+61491570006", "  Ring   me. ")).unwrap();
    assert_eq!(first.id, again.id, "the model repeating itself does not fill the list");
    assert_eq!(store.list(None, "").len(), 1);
}

#[test]
fn three_messages_a_call() {
    let store = Store::in_memory();
    for n in 1..=PER_CALL {
        store.add(new("call_1", "+61491570006", &format!("Message {n}"))).unwrap();
    }
    let refused = store.add(new("call_1", "+61491570006", "Message 4")).unwrap_err();
    assert_eq!((refused.status, refused.code), (429, "call_limit"));
    // Another call of the same caller has its own three (the per-number limit is the day's).
    assert!(store.add(new("call_2", "+61491570006", "Message 5")).is_ok());
    assert_eq!(store.list(None, "").len(), 4);
}

#[test]
fn twenty_a_number_a_day_by_the_last_nine_digits() {
    let store = Store::in_memory();
    for n in 0..PER_CALLER_DAY {
        // The same person, written three ways.
        let number = ["+61491570006", "0491 570 006", "61 491 570 006"][n % 3];
        store.add_at(new(&format!("call_{n}"), number, &format!("Message {n}")), noon() + Duration::minutes(n as i64)).unwrap();
    }
    let refused = store.add_at(new("call_99", "0491570006", "One more"), noon() + Duration::hours(2)).unwrap_err();
    assert_eq!((refused.status, refused.code), (429, "caller_limit"));
    assert!(store.add_at(new("call_100", "+61491570156", "Someone else"), noon() + Duration::hours(2)).is_ok(), "another number is another caller");
    // A day later the number may leave messages again.
    assert!(store.add_at(new("call_101", "0491570006", "Tomorrow"), noon() + Duration::hours(25)).is_ok());
}

/// A message left waiting, put straight in the store (a test of a limit does not need to go through every other one).
fn waiting(store: &Store, n: usize, from: &str, when: chrono::DateTime<Utc>, state: State) -> String {
    let id = format!("msg_w{n}");
    store.lock().messages.push(Message { id: id.clone(), at: when.to_rfc3339(), call_id: format!("call_w{n}"), from: from.into(), name: String::new(), callback: String::new(), message: format!("waiting {n}"), urgency: Urgency::Normal, wants_callback: false, state, seen_at: None, handled_at: (state == State::Handled).then(|| when.to_rfc3339()), handled_by: None });
    id
}

#[test]
fn every_hidden_number_shares_one_small_allowance_a_call_each_does_not_get_its_own() {
    let store = Store::in_memory();
    // Hidden, withheld, made up: none is a number, so all are one caller.
    let hiding = ["", "Private", "anonymous", "12", "unknown", "-1"];
    for (n, from) in hiding.iter().enumerate() {
        store.add_at(new(&format!("call_{n}"), from, &format!("From {n}")), noon() + Duration::minutes(n as i64)).unwrap();
    }
    assert_eq!(PER_WITHHELD_DAY, hiding.len());
    let refused = store.add_at(new("call_new", "", "One more from a new call"), noon() + Duration::hours(1)).unwrap_err();
    assert_eq!((refused.status, refused.code), (429, "withheld_limit"));
    // A caller with a number is not counted against them, and they are not counted against a caller with a number.
    assert!(store.add_at(new("call_known", "+61491570006", "I can be rung"), noon() + Duration::hours(1)).is_ok());
    // A day later the hidden callers may leave messages again.
    assert!(store.add_at(new("call_later", "Private", "Tomorrow"), noon() + Duration::hours(25)).is_ok());
    // The limit of a call still holds inside their share.
    let store = Store::in_memory();
    for n in 0..PER_CALL {
        store.add(new("call_a", "Private", &format!("From a {n}"))).unwrap();
    }
    assert_eq!(store.add(new("call_a", "Private", "Too many")).unwrap_err().code, "call_limit");
}

#[test]
fn hidden_numbers_together_hold_a_small_part_of_the_store_and_only_their_handled_messages_make_room() {
    let store = Store::in_memory();
    for n in 0..MAX_WITHHELD {
        waiting(&store, n, if n % 2 == 0 { "" } else { "Private" }, noon() - Duration::days(2) + Duration::seconds(n as i64), State::New);
    }
    // A number that is waiting on the owner does not make room for a hidden one, and the page says so.
    let refused = store.add_at(new("call_x", "", "No room for me"), noon()).unwrap_err();
    assert_eq!((refused.status, refused.code), (507, "withheld_full"));
    assert_eq!(store.list(None, "").len(), MAX_WITHHELD, "nothing was dropped");
    let notice = store.notice().expect("the Messages page is told");
    assert!(notice.contains("hid their number") && notice.contains("refused") && notice.contains(&MAX_WITHHELD.to_string()), "{notice}");
    // The store has room for everybody else: a caller with a number is kept.
    assert!(store.add_at(new("call_known", "+61491570006", "Ring me"), noon()).is_ok());
    // A handled message of theirs makes room (the oldest handled), and one that is only seen does not.
    let seen = waiting(&store, 900, "", noon() - Duration::days(3), State::Seen);
    assert_eq!(store.add_at(new("call_y", "", "Still no room"), noon()).unwrap_err().code, "withheld_full");
    store.set_state("msg_w4", State::Handled, "owner").unwrap();
    store.set_state("msg_w8", State::Handled, "owner").unwrap();
    assert!(store.add_at(new("call_z", "", "Room now"), noon()).is_ok());
    assert!(store.get("msg_w4").is_none() && store.get("msg_w8").is_some(), "the oldest handled one made room");
    assert!(store.get(&seen).is_some(), "one waiting for the owner is never dropped");
    // And a handled message of a caller with a number does not make room for a hidden one.
    let known = Store::in_memory();
    for n in 0..MAX_WITHHELD {
        waiting(&known, n, "", noon() - Duration::days(2) + Duration::seconds(n as i64), State::New);
    }
    let k = waiting(&known, 500, "+61491570001", noon() - Duration::days(5), State::Handled);
    assert_eq!(known.add_at(new("call_q", "", "Nope"), noon()).unwrap_err().code, "withheld_full");
    assert!(known.get(&k).is_some());
    // Nothing to say while there is room.
    assert_eq!(Store::in_memory().notice(), None);
}

#[test]
fn a_store_full_of_messages_nobody_has_handled_says_so_and_a_number_may_not_have_too_many_waiting() {
    let full = Store::in_memory();
    for n in 0..MAX_STORED {
        waiting(&full, n, &format!("+6140000{n:04}"), noon() - Duration::days(1) + Duration::seconds(n as i64), State::New);
    }
    let notice = full.notice().expect("told");
    assert!(notice.contains(&MAX_STORED.to_string()) && notice.contains("no more can be kept"), "{notice}");
    assert_eq!(full.add_at(new("call_x", "+61491570999", "No room"), noon()).unwrap_err().code, "store_full");

    // One number may not have more than so many waiting, however slowly it leaves them.
    let store = Store::in_memory();
    for n in 0..PER_NUMBER_WAITING {
        waiting(&store, n, "+61491570006", noon() - Duration::days(3) - Duration::minutes(n as i64), if n % 2 == 0 { State::New } else { State::Seen });
    }
    let refused = store.add_at(new("call_z", "0491 570 006", "One more"), noon()).unwrap_err();
    assert_eq!((refused.status, refused.code), (429, "caller_waiting"));
    assert!(store.add_at(new("call_y", "+61491570156", "Someone else"), noon()).is_ok());
    store.set_state("msg_w0", State::Handled, "owner").unwrap();
    assert!(store.add_at(new("call_z", "0491 570 006", "Room again"), noon()).is_ok(), "handling one made room for another");
}

#[test]
fn old_handled_messages_go_and_a_full_store_never_drops_a_message_nobody_has_handled() {
    let store = Store::in_memory();
    let old = store.add_at(new("call_old", "+61491570001", "Old news."), noon() - Duration::days(200)).unwrap();
    let keep = store.add_at(new("call_keep", "+61491570002", "Unread and old."), noon() - Duration::days(200)).unwrap();
    store.set_state(&old.id, State::Handled, "owner").unwrap();
    {
        // Handled long ago (the clock of the state change is now, so age it).
        let mut inner = store.lock();
        for m in inner.messages.iter_mut().filter(|m| m.id == old.id) {
            m.handled_at = Some((noon() - Duration::days(KEEP_HANDLED_DAYS + 1)).to_rfc3339());
        }
    }
    store.add_at(new("call_new", "+61491570003", "New."), noon()).unwrap();
    assert!(store.get(&old.id).is_none(), "handled and older than 90 days: let go");
    assert!(store.get(&keep.id).is_some(), "an unread message is never let go by age");

    // Full of unhandled messages: the next is refused, none is dropped.
    let full = Store::in_memory();
    {
        let mut inner = full.lock();
        for n in 0..MAX_STORED {
            inner.messages.push(Message { id: format!("msg_{n}"), at: (noon() - Duration::days(1) + Duration::seconds(n as i64)).to_rfc3339(), call_id: format!("c{n}"), from: format!("+6140000{n:04}"), name: String::new(), callback: String::new(), message: format!("m{n}"), urgency: Urgency::Normal, wants_callback: false, state: State::New, seen_at: None, handled_at: None, handled_by: None });
        }
    }
    let refused = full.add_at(new("call_x", "+61491570999", "No room"), noon()).unwrap_err();
    assert_eq!((refused.status, refused.code), (507, "store_full"));
    assert_eq!(full.list(None, "").len(), MAX_STORED);
    // One handled message makes room: the oldest handled one.
    full.set_state("msg_5", State::Handled, "owner").unwrap();
    full.set_state("msg_9", State::Handled, "owner").unwrap();
    assert!(full.add_at(new("call_x", "+61491570999", "Room now"), noon()).is_ok());
    assert!(full.get("msg_5").is_none() && full.get("msg_9").is_some(), "the oldest handled message made room");
    assert_eq!(full.list(None, "").len(), MAX_STORED);
}

#[test]
fn states_and_times_follow_the_owner() {
    let store = Store::in_memory();
    let m = store.add(new("call_1", "+61491570006", "Ring me.")).unwrap();
    assert_eq!(store.unread(), 1);
    let seen = store.set_state(&m.id, State::Seen, "owner").unwrap();
    assert!(seen.seen_at.is_some() && seen.handled_at.is_none());
    assert_eq!(store.unread(), 0);
    let handled = store.set_state(&m.id, State::Handled, "owner").unwrap();
    assert_eq!((handled.state, handled.handled_by.as_deref()), (State::Handled, Some("owner")));
    assert_eq!(handled.seen_at, seen.seen_at, "seen keeps the time it was first seen");
    let again = store.set_state(&m.id, State::New, "owner").unwrap();
    assert!(again.seen_at.is_none() && again.handled_at.is_none() && again.handled_by.is_none());
    assert_eq!(store.unread(), 1);
    // Handling a message nobody saw counts as having seen it.
    let direct = store.add(new("call_2", "+61491570006", "Second.")).unwrap();
    assert!(store.set_state(&direct.id, State::Handled, "owner").unwrap().seen_at.is_some());
    assert_eq!(store.set_state("nope", State::Seen, "owner").unwrap_err().status, 404);
    assert_eq!(store.remove("nope").unwrap_err().status, 404);
    store.remove(&m.id).unwrap();
    assert!(store.get(&m.id).is_none());
}

#[test]
fn messages_are_found_by_name_number_or_words_and_listed_newest_first() {
    let store = Store::in_memory();
    store.add_at(NewMessage { name: "Sam".into(), ..new("call_1", "+61491570006", "The gate is jammed.") }, noon()).unwrap();
    store.add_at(NewMessage { name: "Alex".into(), ..new("call_2", "+61491570156", "Please ring about Friday.") }, noon() + Duration::minutes(5)).unwrap();
    let all = store.list(None, "");
    assert_eq!(all.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["Alex", "Sam"]);
    assert_eq!(store.list(None, "gate").len(), 1);
    assert_eq!(store.list(None, "ALEX").len(), 1);
    assert_eq!(store.list(None, "0491 570 156").len(), 1);
    assert_eq!(store.list(None, "+61 491 570 006")[0].name, "Sam");
    assert!(store.list(None, "12").is_empty(), "two digits are not a number to search by");
    assert_eq!(store.list(Some(State::Handled), "").len(), 0);
}

#[test]
fn the_file_is_owner_only_survives_a_restart_and_a_bad_file_is_kept_aside() {
    let dir = TempDir::new("messages-file");
    let store = Store::open(&dir.0);
    let m = store.add(new("call_1", "+61491570006", "Ring me about Friday.")).unwrap();
    store.set_state(&m.id, State::Seen, "owner").unwrap();
    assert_private(&dir.0.join("messages").join(FILE_NAME));
    let again = Store::open(&dir.0);
    assert_eq!(again.get(&m.id).as_ref(), store.get(&m.id).as_ref());
    assert!(!again.was_quarantined());

    let bad = TempDir::new("messages-bad");
    std::fs::create_dir_all(bad.0.join("messages")).unwrap();
    std::fs::write(bad.0.join("messages").join(FILE_NAME), "{not json").unwrap();
    let store = Store::open(&bad.0);
    assert!(store.was_quarantined() && store.list(None, "").is_empty());
    assert_eq!(std::fs::read_to_string(bad.0.join("messages").join("messages.json.corrupt")).unwrap(), "{not json", "the original is never written over");
    // A file from a newer OAIY is not overwritten either.
    let newer = TempDir::new("messages-newer");
    std::fs::create_dir_all(newer.0.join("messages")).unwrap();
    std::fs::write(newer.0.join("messages").join(FILE_NAME), r#"{"version": 9, "messages": []}"#).unwrap();
    assert!(Store::open(&newer.0).was_quarantined());
}

#[test]
fn a_message_that_cannot_be_saved_is_not_kept_and_the_caller_is_told() {
    let dir = TempDir::new("messages-unsaved");
    let store = Store::open(&dir.0);
    // The place the file goes is taken by a folder: the write fails.
    std::fs::create_dir_all(dir.0.join("messages").join(FILE_NAME)).unwrap();
    let e = store.add(new("call_1", "+61491570006", "Ring me.")).unwrap_err();
    assert_eq!((e.status, e.code), (500, "save_failed"));
    assert!(store.list(None, "").is_empty(), "so it is not listed as if it were kept: the receptionist must not say it was");
}

struct Notifier(std::sync::Mutex<Vec<String>>);

impl MessageNotifier for Notifier {
    fn message_taken(&self, m: &Message) -> bool {
        self.0.lock().unwrap().push(m.id.clone());
        true
    }
}

#[test]
fn the_owner_is_told_only_when_something_can_tell_them() {
    // The notifier is process-wide: this test is the only one that sets it.
    let m = Store::in_memory().add(new("call_n", "+61491570006", "Ring me.")).unwrap();
    set_notifier(None);
    assert!(!notify(&m), "nothing to tell them with: not notified");
    let fake = Arc::new(Notifier(Default::default()));
    set_notifier(Some(fake.clone()));
    assert!(notify(&m));
    assert!(fake.0.lock().unwrap().contains(&m.id));
    set_notifier(None);
    assert!(!notify(&m));
}
