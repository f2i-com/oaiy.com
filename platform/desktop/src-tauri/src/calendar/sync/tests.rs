//! The sync against a FormLogic that is there, one that is not, and one from
//! before `updatedSince`: a fake FormLogic on a local port, speaking the same
//! `/api/v1` the real one does, and a closed port for "unreachable".

use super::*;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use chrono::Timelike;

#[derive(Clone, Debug)]
struct Rec {
    id: String,
    answers: Value,
    updated: DateTime<Utc>,
    /// Bumped on every write: the etag.
    version: u64,
    born: u64,
}

/// A FormLogic with one appointments form.
struct Fake {
    form: String,
    /// Answers as a FormLogic from before `updatedSince`, tombstones and etags.
    old_api: bool,
    revoked: bool,
    clock: DateTime<Utc>,
    counter: u64,
    records: Vec<Rec>,
    tombstones: Vec<(String, DateTime<Utc>)>,
    keys: HashMap<String, (String, String)>,
    log: Vec<String>,
    /// Make the next record POSTed, and drop the connection without answering.
    drop_next_create_reply: bool,
    /// Someone changes this record in FormLogic just before the next PUT reaches it.
    edit_before_put: Option<(String, &'static str, &'static str)>,
    /// Refuse (400) a record whose notes contain this.
    refuse_notes: Option<&'static str>,
}

fn fmt(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d %H:%M:%S").to_string()
}

impl Fake {
    fn new() -> Self {
        Self {
            form: "F1".into(),
            old_api: false,
            revoked: false,
            clock: Utc::now().with_nanosecond(0).unwrap(),
            counter: 0,
            records: Vec::new(),
            tombstones: Vec::new(),
            keys: HashMap::new(),
            log: Vec::new(),
            drop_next_create_reply: false,
            edit_before_put: None,
            refuse_notes: None,
        }
    }

    /// The time of a write: a second after the last, so each is told apart.
    fn tick(&mut self) -> DateTime<Utc> {
        self.clock += chrono::Duration::seconds(1);
        self.counter += 1;
        self.clock
    }

    fn json(&self, r: &Rec) -> Value {
        let mut v = json!({"id": r.id, "answers": r.answers, "status": "submitted", "updatedAt": fmt(r.updated), "submittedAt": fmt(r.updated)});
        if !self.old_api {
            v["etag"] = json!(format!("e{}", r.version));
        }
        v
    }

    /// A record made in FormLogic (by a person or a flow).
    fn add(&mut self, answers: Value) -> String {
        let t = self.tick();
        let id = format!("r{}", self.counter);
        self.records.push(Rec { id: id.clone(), answers, updated: t, version: self.counter, born: self.counter });
        id
    }

    fn edit(&mut self, id: &str, field: &str, value: &str) {
        let t = self.tick();
        let v = self.counter;
        let r = self.records.iter_mut().find(|r| r.id == id).expect("no such record");
        r.answers[field] = json!(value);
        r.updated = t;
        r.version = v;
    }

    fn delete(&mut self, id: &str) {
        let t = self.tick();
        self.records.retain(|r| r.id != id);
        self.tombstones.push((id.to_string(), t));
    }

    fn get(&self, id: &str) -> &Rec {
        self.records.iter().find(|r| r.id == id).expect("no such record")
    }

    fn by_key(&self, key: &str) -> Vec<&Rec> {
        self.records.iter().filter(|r| r.answers["request_id"] == json!(key)).collect()
    }

    fn count(&self, what: &str) -> usize {
        self.log.iter().filter(|l| l.starts_with(what)).count()
    }

    fn handle(&mut self, method: &str, target: &str, headers: &str, body: &str) -> Option<(u16, Value)> {
        self.log.push(format!("{method} {target}"));
        if self.revoked {
            return Some((401, json!({"error": true, "message": "Invalid API key"})));
        }
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let q: Vec<(String, String)> = query.split('&').filter(|p| !p.is_empty()).map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k), decode(v))
        }).collect();
        let arg = |k: &str| q.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        let base = format!("/api/v1/forms/{}/responses", self.form);
        if method == "GET" && path == "/api/v1/app-logic" {
            return Some((200, json!({"apps": [{"slug": "aokie", "forms": [{"packFormId": "customers", "formId": "C1"}, {"packFormId": "appointments", "formId": self.form}]}]})));
        }
        if path == base {
            return match method {
                "GET" => Some(self.list(&arg)),
                "POST" => self.create(body),
                _ => Some((405, json!({}))),
            };
        }
        let Some(id) = path.strip_prefix(&format!("{base}/")) else { return Some((404, json!({"error": true, "message": "Form not found or access denied"}))) };
        let id = decode(id);
        match method {
            "GET" => Some(match self.records.iter().find(|r| r.id == id) {
                Some(r) => (200, json!({"response": self.json(r)})),
                None => (404, json!({"error": true, "message": "Response not found"})),
            }),
            "PUT" => Some(self.update(&id, headers, body)),
            "DELETE" => Some(if self.records.iter().any(|r| r.id == id) {
                self.delete(&id);
                (200, json!({"success": true}))
            } else {
                (404, json!({"error": true, "message": "Response not found"}))
            }),
            _ => Some((405, json!({}))),
        }
    }

    fn list(&self, arg: &dyn Fn(&str) -> Option<String>) -> (u16, Value) {
        let key = arg("answers.request_id");
        let mut recs: Vec<&Rec> = self.records.iter().filter(|r| key.as_ref().map_or(true, |k| r.answers["request_id"] == json!(k))).collect();
        let limit: usize = arg("limit").and_then(|l| l.parse().ok()).unwrap_or(50);
        if let (false, Some(since)) = (self.old_api, arg("updatedSince")) {
            let since = NaiveDateTime::parse_from_str(&since, "%Y-%m-%d %H:%M:%S").unwrap().and_utc();
            let after = arg("afterId");
            recs.retain(|r| match &after {
                Some(a) => r.updated > since || (r.updated == since && r.id > *a),
                None => r.updated >= since,
            });
            recs.sort_by(|a, b| (a.updated, &a.id).cmp(&(b.updated, &b.id)));
            recs.truncate(limit);
            let deleted: Vec<Value> = self.tombstones.iter().filter(|(_, t)| *t >= since).map(|(id, t)| json!({"id": id, "deletedAt": fmt(*t)})).collect();
            return (200, json!({"responses": recs.iter().map(|r| self.json(r)).collect::<Vec<_>>(), "deleted": deleted, "deletedSince": EPOCH, "deletedComplete": true, "serverTime": fmt(self.clock)}));
        }
        // Newest first, by offset, as FormLogic lists without updatedSince.
        recs.sort_by(|a, b| b.born.cmp(&a.born));
        let offset: usize = arg("offset").and_then(|o| o.parse().ok()).unwrap_or(0);
        let page: Vec<Value> = recs.into_iter().skip(offset).take(limit).map(|r| self.json(r)).collect();
        (200, json!({"responses": page}))
    }

    fn create(&mut self, body: &str) -> Option<(u16, Value)> {
        let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        let answers = body["answers"].clone();
        // Scoped to the form, as FormLogic's submission ledger is.
        let key = body["idempotencyKey"].as_str().map(|k| format!("{}:{k}", self.form));
        if let Some(k) = &key {
            if let Some((sent, id)) = self.keys.get(k) {
                return Some(if *sent == answers.to_string() {
                    (200, json!({"response": {"id": id}, "idempotent": true}))
                } else {
                    (409, json!({"error": true, "conflict": true, "message": "This idempotency key was already used with a different submission."}))
                });
            }
        }
        if answers["service"].as_str().unwrap_or("").is_empty() {
            return Some((400, json!({"error": true, "message": "Validation failed"})));
        }
        if let Some(bad) = self.refuse_notes {
            if answers["notes"].as_str().unwrap_or("").contains(bad) {
                return Some((400, json!({"error": true, "message": "Validation failed: notes"})));
            }
        }
        let id = self.add(answers.clone());
        if let Some(k) = key {
            self.keys.insert(k, (answers.to_string(), id.clone()));
        }
        if std::mem::take(&mut self.drop_next_create_reply) {
            return None;
        }
        let r = self.get(&id).clone();
        // FormLogic answers a create without an etag.
        Some((201, json!({"response": {"id": r.id, "answers": r.answers, "status": "submitted", "updatedAt": fmt(r.updated)}})))
    }

    fn update(&mut self, id: &str, headers: &str, body: &str) -> (u16, Value) {
        if !self.records.iter().any(|r| r.id == id) {
            return (404, json!({"error": true, "message": "Response not found"}));
        }
        if let Some((who, field, value)) = self.edit_before_put.clone() {
            if who == id {
                self.edit_before_put = None;
                self.edit(id, field, value);
            }
        }
        let if_match = headers.lines().find_map(|l| l.strip_prefix("if-match: ").or_else(|| l.strip_prefix("If-Match: "))).map(|v| v.trim().trim_matches('"').to_string());
        let current = self.get(id).clone();
        if let (false, Some(tag)) = (self.old_api, if_match) {
            if tag != format!("e{}", current.version) {
                return (412, json!({"error": true, "code": "version_conflict", "response": self.json(&current)}));
            }
        }
        let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        if let Some(bad) = self.refuse_notes {
            if body["answers"]["notes"].as_str().unwrap_or("").contains(bad) {
                return (400, json!({"error": true, "message": "Validation failed: notes"}));
            }
        }
        let t = self.tick();
        let v = self.counter;
        let r = self.records.iter_mut().find(|r| r.id == id).unwrap();
        if let (Some(into), Some(from)) = (r.answers.as_object_mut(), body["answers"].as_object()) {
            for (k, val) in from {
                into.insert(k.clone(), val.clone());
            }
        }
        r.updated = t;
        r.version = v;
        let r = r.clone();
        (200, json!({"response": self.json(&r)}))
    }
}

fn decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(b'?'));
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Serve `fake` on a local port: its base URL.
fn serve(fake: Arc<StdMutex<Fake>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut raw = Vec::new();
            let mut buf = [0u8; 8192];
            let (head, body) = loop {
                let Ok(n) = stream.read(&mut buf) else { break (String::new(), String::new()) };
                if n == 0 {
                    break (String::new(), String::new());
                }
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(end) = text.find("\r\n\r\n") {
                    let len: usize = text[..end].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length: ").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
                    if raw.len() >= end + 4 + len {
                        break (text[..end].to_string(), String::from_utf8_lossy(&raw[end + 4..end + 4 + len]).to_string());
                    }
                }
            };
            let mut line = head.lines().next().unwrap_or("").split(' ');
            let (method, target) = (line.next().unwrap_or("").to_string(), line.next().unwrap_or("").to_string());
            let reply = fake.lock().unwrap().handle(&method, &target, &head, &body);
            if let Some((status, v)) = reply {
                let text = v.to_string();
                let _ = write!(stream, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
                let _ = stream.flush();
            }
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn formlogic() -> (Arc<StdMutex<Fake>>, String) {
    let f = Arc::new(StdMutex::new(Fake::new()));
    let base = serve(f.clone());
    (f, base)
}

/// A port nothing listens on.
fn closed() -> String {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    format!("http://127.0.0.1:{port}")
}

fn calendar() -> (Calendar, PathBuf) {
    let dir = std::env::temp_dir().join(format!("oaiy-calsync-{}", uuid::Uuid::new_v4().simple()));
    (Calendar::open(&dir, None), dir)
}

/// A sync, as it runs ten minutes from now (a call's grace has passed).
fn run(cal: &Calendar, base: &str) -> Result<Outcome, Failure> {
    run_with(cal, base, HashSet::new(), chrono::Duration::minutes(10))
}

fn run_with(cal: &Calendar, base: &str, waiting: HashSet<String>, later: chrono::Duration) -> Result<Outcome, Failure> {
    sync(cal, &Api::new(base, "flk_test").unwrap(), &Ctx { now: Utc::now() + later, waiting })
}

fn pending_of(cal: &Calendar) -> Pending {
    pending(&cal.book.lock().unwrap())
}

fn remote_id(cal: &Calendar, id: &str) -> Option<String> {
    cal.get(id).and_then(|a| remote(&a).id)
}

fn new_appt(service: &str, start: &str, name: &str) -> NewAppointment {
    NewAppointment { service: service.into(), start: start.into(), name: name.into(), status: Some(Status::Confirmed), ..Default::default() }
}

fn formlogic_answers(service: &str, date: &str, time: &str, name: &str) -> Value {
    json!({"caller_name": name, "service": service, "date": date, "time": time, "status": "requested", "phone": "", "notes": "", "source": "manual", "call_id": "", "request_id": ""})
}

#[test]
fn a_change_made_while_formlogic_is_unreachable_is_kept_and_sent_when_it_is_back() {
    let (cal, dir) = calendar();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();

    let offline = run(&cal, &closed());
    match offline {
        Err(Failure::Offline(m)) => assert!(m.contains("can't be reached"), "{m}"),
        other => panic!("expected offline, got {other:?}"),
    }
    assert_eq!(cal.list(None, None).len(), 1, "the appointment is kept");
    assert_eq!(pending_of(&cal), Pending { creates: 1, updates: 0, deletes: 0, total: 1 });

    let (fake, base) = formlogic();
    let out = run(&cal, &base).unwrap();
    assert_eq!(out.pushed, 1);
    {
        let f = fake.lock().unwrap();
        assert_eq!(f.records.len(), 1);
        assert_eq!(f.records[0].answers["caller_name"], "Sam");
        assert_eq!(f.records[0].answers["request_id"], json!(format!("oaiy:{}", a.id)), "findable again by its key");
    }
    assert_eq!(pending_of(&cal).total, 0);

    // Nothing goes twice.
    let again = run(&cal, &base).unwrap();
    assert_eq!((again.pushed, again.pulled), (0, 0));
    assert_eq!(fake.lock().unwrap().records.len(), 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn changes_on_both_sides_while_apart_all_arrive() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    let b = fake.lock().unwrap().add(formlogic_answers("Colour", "2026-10-03", "10:00", "Ali"));
    run(&cal, &base).unwrap();
    let ra = remote_id(&cal, &a.id).unwrap();

    // Apart: both sides change.
    cal.update(&a.id, Change { notes: Some("gate code 1234".into()), ..Default::default() }).unwrap();
    let d = cal.create(new_appt("Trim", "2026-10-04T09:00", "Dee")).unwrap();
    {
        let mut f = fake.lock().unwrap();
        f.edit(&b, "status", "confirmed");
        f.add(formlogic_answers("Wash", "2026-10-05", "11:30", "Cy"));
    }
    assert!(matches!(run(&cal, &closed()), Err(Failure::Offline(_))));
    assert_eq!(pending_of(&cal), Pending { creates: 1, updates: 1, deletes: 0, total: 2 });

    let out = run(&cal, &base).unwrap();
    assert_eq!((out.pushed, out.pulled), (2, 2), "{out:?}");
    let f = fake.lock().unwrap();
    assert_eq!(f.get(&ra).answers["notes"], "gate code 1234");
    assert_eq!(f.by_key(&format!("oaiy:{}", d.id)).len(), 1);
    assert_eq!(f.records.len(), 4);
    drop(f);
    let here = cal.list(None, None);
    assert_eq!(here.len(), 4);
    assert_eq!(here.iter().find(|x| x.name == "Ali").unwrap().status, Status::Confirmed);
    assert!(here.iter().any(|x| x.name == "Cy" && x.start == "2026-10-05T11:30" && x.source == "formlogic"));
    assert_eq!(pending_of(&cal).total, 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_appointment_deleted_here_is_deleted_in_formlogic_and_does_not_come_back() {
    for old_api in [false, true] {
        let (cal, dir) = calendar();
        let (fake, base) = formlogic();
        fake.lock().unwrap().old_api = old_api;
        let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
        run(&cal, &base).unwrap();
        let ra = remote_id(&cal, &a.id).unwrap();

        cal.remove(&a.id).unwrap();
        assert_eq!(pending_of(&cal).deletes, 1, "the deletion waits to be sent");
        assert!(matches!(run(&cal, &closed()), Err(Failure::Offline(_))));
        assert_eq!(pending_of(&cal).deletes, 1, "and is kept while FormLogic is away");

        run(&cal, &base).unwrap();
        assert!(fake.lock().unwrap().records.is_empty(), "old_api={old_api}");
        assert_eq!(fake.lock().unwrap().count(&format!("DELETE /api/v1/forms/F1/responses/{ra}")), 1);
        run(&cal, &base).unwrap();
        assert!(cal.list(None, None).is_empty(), "not brought back (old_api={old_api})");
        assert_eq!(pending_of(&cal).total, 0);
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn an_appointment_deleted_in_formlogic_is_deleted_here() {
    for old_api in [false, true] {
        let (cal, dir) = calendar();
        let (fake, base) = formlogic();
        fake.lock().unwrap().old_api = old_api;
        let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
        let keep = cal.create(new_appt("Trim", "2026-10-03T14:00", "Kim")).unwrap();
        run(&cal, &base).unwrap();
        let ra = remote_id(&cal, &a.id).unwrap();

        fake.lock().unwrap().delete(&ra);
        let out = run(&cal, &base).unwrap();
        assert_eq!(out.removed, 1, "old_api={old_api}");
        assert!(cal.get(&a.id).is_none());
        assert!(cal.get(&keep.id).is_some(), "the rest stay (old_api={old_api})");
        assert_eq!(fake.lock().unwrap().records.len(), 1, "and nothing is sent back");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_deletion_on_one_side_wins_over_an_edit_on_the_other() {
    // Edited here, deleted there.
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    run(&cal, &base).unwrap();
    let ra = remote_id(&cal, &a.id).unwrap();
    cal.update(&a.id, Change { notes: Some("late".into()), ..Default::default() }).unwrap();
    fake.lock().unwrap().delete(&ra);
    run(&cal, &base).unwrap();
    assert!(cal.get(&a.id).is_none());
    assert!(fake.lock().unwrap().records.is_empty(), "not made again");
    let _ = std::fs::remove_dir_all(dir);

    // Deleted here, edited there.
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    run(&cal, &base).unwrap();
    let ra = remote_id(&cal, &a.id).unwrap();
    cal.remove(&a.id).unwrap();
    fake.lock().unwrap().edit(&ra, "status", "confirmed");
    run(&cal, &base).unwrap();
    assert!(fake.lock().unwrap().records.is_empty());
    assert!(cal.list(None, None).is_empty(), "not brought back");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn when_both_sides_change_it_each_keeps_its_own_fields() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    run(&cal, &base).unwrap();
    let ra = remote_id(&cal, &a.id).unwrap();

    // Cancelled here; moved there. Neither is lost.
    cal.update(&a.id, Change { status: Some(Status::Cancelled), notes: Some("called to cancel".into()), ..Default::default() }).unwrap();
    {
        let mut f = fake.lock().unwrap();
        f.edit(&ra, "time", "16:00");
        f.edit(&ra, "phone", "0400000001");
    }
    run(&cal, &base).unwrap();
    let here = cal.get(&a.id).unwrap();
    let f = fake.lock().unwrap();
    let there = &f.get(&ra).answers;
    assert_eq!((here.status, here.start.as_str(), here.phone.as_str(), here.notes.as_str()), (Status::Cancelled, "2026-10-02T16:00", "0400000001", "called to cancel"));
    assert_eq!((there["status"].as_str(), there["time"].as_str(), there["phone"].as_str(), there["notes"].as_str()), (Some("cancelled"), Some("16:00"), Some("0400000001"), Some("called to cancel")));
    drop(f);
    assert_eq!(pending_of(&cal).total, 0, "the two sides agree");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn when_both_sides_change_the_same_field_a_final_status_wins_then_the_later_change() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    let b = cal.create(new_appt("Trim", "2026-10-03T14:00", "Kim")).unwrap();
    run(&cal, &base).unwrap();
    let (ra, rb) = (remote_id(&cal, &a.id).unwrap(), remote_id(&cal, &b.id).unwrap());

    // Cancelled here, confirmed there later: cancelled is final, and stands.
    cal.update(&a.id, Change { status: Some(Status::Cancelled), ..Default::default() }).unwrap();
    // Moved here; moved there an hour later: FormLogic's time is the later change.
    cal.update(&b.id, Change { start: Some("2026-10-03T15:00".into()), ..Default::default() }).unwrap();
    {
        let mut f = fake.lock().unwrap();
        f.clock = Utc::now() + chrono::Duration::hours(1);
        f.edit(&ra, "status", "confirmed");
        f.edit(&rb, "time", "16:00");
    }
    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(cal.get(&a.id).unwrap().status, Status::Cancelled);
    assert_eq!(f.get(&ra).answers["status"], "cancelled");
    assert_eq!(cal.get(&b.id).unwrap().start, "2026-10-03T16:00");
    assert_eq!(f.get(&rb).answers["time"], "16:00");
    drop(f);
    assert_eq!(pending_of(&cal).total, 0);

    // And the other way: moved here an hour after it was moved there.
    cal.update(&b.id, Change { start: Some("2026-10-03T17:00".into()), ..Default::default() }).unwrap();
    {
        let mut f = fake.lock().unwrap();
        f.clock = Utc::now() - chrono::Duration::hours(1);
        f.edit(&rb, "time", "08:00");
    }
    run(&cal, &base).unwrap();
    assert_eq!(cal.get(&b.id).unwrap().start, "2026-10-03T17:00");
    assert_eq!(fake.lock().unwrap().get(&rb).answers["time"], "17:00");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_write_that_meets_a_change_made_in_formlogic_meanwhile_is_merged_not_written_over_it() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    run(&cal, &base).unwrap();
    let ra = remote_id(&cal, &a.id).unwrap();

    cal.update(&a.id, Change { notes: Some("gate code".into()), ..Default::default() }).unwrap();
    {
        // Between this desktop's read and its write, someone cancels it in FormLogic.
        let mut f = fake.lock().unwrap();
        f.edit_before_put = Some((ra.clone(), "status", "cancelled"));
    }
    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(f.count(&format!("PUT /api/v1/forms/F1/responses/{ra}")), 2, "refused once (412), then sent over the new version");
    assert_eq!(f.get(&ra).answers["status"], "cancelled", "the cancellation made meanwhile stands");
    assert_eq!(f.get(&ra).answers["notes"], "gate code", "and so does the note made here");
    drop(f);
    assert_eq!(cal.get(&a.id).unwrap().status, Status::Cancelled, "both reach this desktop");
    assert_eq!(cal.get(&a.id).unwrap().notes, "gate code");
    assert_eq!(pending_of(&cal).total, 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_record_formlogic_refuses_is_set_aside_and_the_rest_still_go() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    fake.lock().unwrap().refuse_notes = Some("REFUSE");
    let bad = cal.create(NewAppointment { notes: "REFUSE this".into(), ..new_appt("Cut", "2026-10-02T14:00", "Sam") }).unwrap();
    let good = cal.create(new_appt("Trim", "2026-10-03T14:00", "Kim")).unwrap();

    let out = run(&cal, &base).unwrap();
    assert_eq!(out.pushed, 1);
    assert_eq!(out.problems.len(), 1);
    assert_eq!(out.problems[0].id, bad.id);
    assert!(out.problems[0].message.contains("Validation failed"), "{:?}", out.problems);
    assert!(remote_id(&cal, &good.id).is_some());
    assert_eq!(pending_of(&cal).total, 0, "a refused record is not counted as waiting");

    // Not sent again as it is...
    let posts = fake.lock().unwrap().count("POST ");
    run(&cal, &base).unwrap();
    assert_eq!(fake.lock().unwrap().count("POST "), posts);
    // ...but once it changes.
    cal.update(&bad.id, Change { notes: Some("fine now".into()), ..Default::default() }).unwrap();
    run(&cal, &base).unwrap();
    assert!(remote_id(&cal, &bad.id).is_some());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_call_without_a_service_is_still_taken() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    cal.create(NewAppointment { service: String::new(), ..new_appt("", "2026-10-02T14:00", "Sam") }).unwrap();
    let out = run(&cal, &base).unwrap();
    assert_eq!((out.pushed, out.problems.len()), (1, 0));
    assert_eq!(fake.lock().unwrap().records[0].answers["service"], "Appointment");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_create_whose_answer_was_lost_is_found_again_not_made_twice() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    fake.lock().unwrap().drop_next_create_reply = true;
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    assert!(matches!(run(&cal, &base), Err(Failure::Offline(_))), "the answer never came");
    assert_eq!(fake.lock().unwrap().records.len(), 1, "but FormLogic made it");

    // Changed before the next try, so a replay would not even match.
    cal.update(&a.id, Change { notes: Some("changed".into()), ..Default::default() }).unwrap();
    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(f.records.len(), 1, "one record");
    assert_eq!(remote_id(&cal, &a.id).as_deref(), Some(f.records[0].id.as_str()));
    assert_eq!(f.records[0].answers["notes"], "changed");
    drop(f);
    // Deleted here now, it goes there too.
    cal.remove(&a.id).unwrap();
    run(&cal, &base).unwrap();
    assert!(fake.lock().unwrap().records.is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

fn call(request: &str) -> Value {
    json!({"requestId": request, "callId": "call-1", "from": "0491570006", "callerName": "Lance", "service": "Lawn mowing", "date": "2026-10-01", "time": "10:00"})
}

fn flow_record(request: &str) -> Value {
    json!({"caller_name": "Lance", "service": "Lawn mowing", "date": "2026-10-01", "time": "10:00", "status": "requested", "phone": "0491570006", "notes": "Asked on the call for a Thursday morning.", "source": "call", "call_id": "call-1", "request_id": request})
}

#[test]
fn a_call_recorded_while_offline_pairs_with_formlogics_own_record_of_it() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.record_request(&call("req-1")).unwrap();
    assert!(matches!(run(&cal, &closed()), Err(Failure::Offline(_))));
    // Back online, the event reached FormLogic and its flow recorded the request.
    let flows = fake.lock().unwrap().add(flow_record("req-1"));

    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(f.records.len(), 1, "no second record");
    assert_eq!(f.count("POST "), 0);
    drop(f);
    assert_eq!(cal.list(None, None).len(), 1, "no second appointment here");
    assert_eq!(remote_id(&cal, &a.id).as_deref(), Some(flows.as_str()));
    assert_eq!(cal.get(&a.id).unwrap().notes, "Asked on the call for a Thursday morning.", "FormLogic's record is taken in");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn when_both_recorded_one_call_formlogics_record_is_kept_and_the_desktops_deleted() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.record_request(&call("req-2")).unwrap();
    run(&cal, &base).unwrap();
    let ours = remote_id(&cal, &a.id).unwrap();
    // The event arrived late and FormLogic's flow made its own anyway.
    let flows = fake.lock().unwrap().add(flow_record("req-2"));

    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(f.records.iter().map(|r| r.id.clone()).collect::<Vec<_>>(), vec![flows.clone()], "the desktop's copy is gone");
    assert!(f.count(&format!("DELETE /api/v1/forms/F1/responses/{ours}")) == 1);
    drop(f);
    assert_eq!(cal.list(None, None).len(), 1);
    assert_eq!(remote_id(&cal, &a.id).as_deref(), Some(flows.as_str()));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_call_waits_for_formlogics_own_record_while_its_event_is_on_the_way() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    cal.record_request(&call("req-3")).unwrap();
    // Within five minutes of the call.
    run_with(&cal, &base, HashSet::new(), chrono::Duration::zero()).unwrap();
    // Later, but its event has still not reached FormLogic.
    run_with(&cal, &base, HashSet::from(["req-3".to_string()]), chrono::Duration::minutes(30)).unwrap();
    assert_eq!(fake.lock().unwrap().count("POST "), 0);
    assert_eq!(pending_of(&cal).creates, 1, "and it is shown as waiting");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_call_deleted_here_is_not_brought_back_by_its_event_or_formlogics_flow() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.record_request(&call("req-4")).unwrap();
    cal.remove(&a.id).unwrap();
    assert!(cal.record_request(&call("req-4")).is_none(), "the same event again");
    // FormLogic's flow records it after all.
    fake.lock().unwrap().add(flow_record("req-4"));
    run(&cal, &base).unwrap();
    assert!(cal.list(None, None).is_empty());
    assert!(fake.lock().unwrap().records.is_empty(), "FormLogic's copy is deleted too");
    // And one that turns up later still.
    fake.lock().unwrap().add(flow_record("req-4"));
    run(&cal, &base).unwrap();
    assert!(cal.list(None, None).is_empty());
    assert!(fake.lock().unwrap().records.is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn after_the_first_sync_only_what_changed_is_read() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    fake.lock().unwrap().add(formlogic_answers("Colour", "2026-10-03", "10:00", "Ali"));
    run(&cal, &base).unwrap();
    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    let lists: Vec<&String> = f.log.iter().filter(|l| l.starts_with("GET /api/v1/forms/F1/responses?updatedSince=")).collect();
    assert_eq!(lists.len(), 2);
    assert!(lists[0].contains("1970-01-01"), "{}", lists[0]);
    assert!(!lists[1].contains("1970-01-01"), "the second asks from the first: {}", lists[1]);
    assert_eq!(f.count("GET /api/v1/forms/F1/responses/"), 0, "and looks nothing up one by one");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_older_formlogic_is_read_in_full_page_by_page() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    {
        let mut f = fake.lock().unwrap();
        f.old_api = true;
        for i in 0..(PAGE + 3) {
            f.add(formlogic_answers("Cut", "2026-10-02", "09:00", &format!("P{i}")));
        }
    }
    let out = run(&cal, &base).unwrap();
    assert_eq!(out.pulled, PAGE + 3);
    assert_eq!(cal.list(None, None).len(), PAGE + 3);
    assert!(fake.lock().unwrap().log.iter().any(|l| l.contains(&format!("offset={PAGE}"))));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_form_is_found_again_after_the_pack_is_installed_again() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    run(&cal, &base).unwrap();
    {
        let mut f = fake.lock().unwrap();
        f.form = "F2".into();
        f.records.clear();
    }
    run(&cal, &base).unwrap();
    let f = fake.lock().unwrap();
    assert_eq!(f.records.len(), 1, "sent to the new form");
    assert_eq!(f.records[0].answers["caller_name"], "Sam");
    drop(f);
    assert!(cal.get(&a.id).is_some(), "and not taken for deleted");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_revoked_key_is_an_error_to_act_on_not_offline() {
    let (cal, dir) = calendar();
    let (fake, base) = formlogic();
    fake.lock().unwrap().revoked = true;
    cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    match run(&cal, &base) {
        Err(Failure::Refused(m)) => assert!(m.contains("link it again"), "{m}"),
        other => panic!("expected refused, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn pending_counts_what_waits_to_go() {
    let (cal, dir) = calendar();
    let (_fake, base) = formlogic();
    let a = cal.create(new_appt("Cut", "2026-10-02T14:00", "Sam")).unwrap();
    let b = cal.create(new_appt("Trim", "2026-10-03T14:00", "Kim")).unwrap();
    run(&cal, &base).unwrap();
    cal.update(&a.id, Change { notes: Some("x".into()), ..Default::default() }).unwrap();
    cal.remove(&b.id).unwrap();
    cal.create(new_appt("Wash", "2026-10-04T14:00", "Lu")).unwrap();
    // Deleted before it ever went: nothing to tell FormLogic.
    let never = cal.create(new_appt("Dry", "2026-10-05T14:00", "Mo")).unwrap();
    cal.remove(&never.id).unwrap();
    assert_eq!(pending_of(&cal), Pending { creates: 1, updates: 1, deletes: 1, total: 3 });
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn offline_tries_wait_longer_each_time_up_to_ten_minutes() {
    assert_eq!(backoff(1), Duration::from_secs(60));
    assert_eq!(backoff(2), Duration::from_secs(120));
    assert_eq!(backoff(4), Duration::from_secs(480));
    assert_eq!(backoff(5), BACKOFF_MAX);
    assert_eq!(backoff(40), BACKOFF_MAX);
}

#[test]
fn a_calendar_from_before_tombstones_still_loads() {
    let dir = std::env::temp_dir().join(format!("oaiy-calsync-old-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("calendar.json"),
        r#"{"settings":{"business":"","hours":[[],[],[],[],[],[],[]],"services":[],"slotMinutes":30,"noticeMinutes":60,"horizonDays":30,"textConfirmations":true},
            "appointments":[{"id":"appt_1","service":"Cut","start":"2026-10-02T14:00","minutes":30,"status":"confirmed","name":"Sam","phone":"","notes":"","source":"manual","createdAt":"2026-09-28T09:00:00Z","updatedAt":"2026-09-28T09:00:00Z","formlogic":{"id":"r9","updatedAt":"2026-09-28 09:00:01","syncedAt":"2026-09-28T09:00:00Z"}}]}"#,
    )
    .unwrap();
    let cal = Calendar::open(&dir, None);
    let a = cal.get("appt_1").expect("the old calendar loads");
    let r = remote(&a);
    assert_eq!((r.id.as_deref(), r.synced_at.as_str()), (Some("r9"), "2026-09-28T09:00:00Z"));
    assert_eq!(pending_of(&cal).total, 0, "in step, as it was");

    // A deletion made offline survives a restart.
    cal.remove("appt_1").unwrap();
    drop(cal);
    let reopened = Calendar::open(&dir, None);
    assert_eq!(pending_of(&reopened), Pending { creates: 0, updates: 0, deletes: 1, total: 1 });
    assert_eq!(reopened.book.lock().unwrap().deleted[0].formlogic_id.as_deref(), Some("r9"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_appointments_form_is_found_in_the_listing() {
    let apps = json!({"apps": [{"id": "x", "forms": [{"packFormId": "customers", "formId": "c1"}, {"packFormId": "appointments", "formId": "cd1d"}]}]});
    assert_eq!(find_form(&apps).as_deref(), Some("cd1d"));
    assert_eq!(find_form(&json!({"apps": []})), None);
}

#[test]
fn statuses_and_sources_go_both_ways() {
    for s in [Status::Requested, Status::Confirmed, Status::Done, Status::Cancelled] {
        assert_eq!(status_here(status_there(s)), s);
    }
    assert_eq!(status_there(Status::Declined), "cancelled");
    assert_eq!(status_here("no-show"), Status::Cancelled);
    assert_eq!((source_there("text"), source_here("sms")), ("sms", "text"));
}

#[test]
fn a_record_becomes_a_change_and_an_appointment_its_answers() {
    let c = change_from(&json!({"caller_name": "Lanes", "service": "Lawnmowing", "date": "2026-10-01", "time": "10:00", "status": "requested", "phone": "0491570006"}));
    assert_eq!((c.start.as_deref(), c.status, c.name.as_deref()), (Some("2026-10-01T10:00"), Some(Status::Requested), Some("Lanes")));
    assert_eq!(change_from(&json!({"date": "2026-10-01", "time": "10:00:00"})).start.as_deref(), Some("2026-10-01T10:00"));
    let (cal, dir) = calendar();
    let a = cal.create(NewAppointment { service: "Cut".into(), start: "2026-10-02T14:30".into(), name: "Sam".into(), source: "text".into(), status: Some(Status::Confirmed), ..Default::default() }).unwrap();
    let v = answers(&a);
    assert_eq!((v["date"].as_str(), v["time"].as_str(), v["status"].as_str(), v["source"].as_str(), v["caller_name"].as_str()), (Some("2026-10-02"), Some("14:30"), Some("confirmed"), Some("sms"), Some("Sam")));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn times_from_both_sides_compare() {
    assert!(when("2026-09-28 09:21:46") < when("2026-09-28T09:21:47Z"));
    assert!(when("nonsense").is_none());
}
