//! The rendezvous of the stub relay (README 10.1, the table of "The rendezvous as the relay serves it"): the five routes, the state machine and what the real relay checks of an offer, a
//! response and an approval (Interpretations 25 to 35 and 50 to 51 in the parts a client depends on).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::b64;
use crate::client::http::HttpRequest;
use crate::ids;
use crate::json::{self, Json};
use crate::keys::{thumbprint_of, VerifyKey, X25519Public};
use crate::pairing::math;

use super::stub::{err, ok, Resp, StubRelay};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PState {
    Open,
    Answered,
    Approved,
    Denied,
    Ended,
}

/// A rendezvous.
pub(crate) struct Pairing {
    desktop: String,
    app_id: String,
    offer: String,
    mac: String,
    desktop_ed: VerifyKey,
    desktop_thumb: String,
    exp: i64,
    state: PState,
    response: Option<String>,
    responses: u32,
    rejects: u32,
    phone_dev: Option<String>,
    sealed: Option<String>,
    receipt: Option<(u64, String, Vec<String>)>,
}

fn not_found() -> Resp {
    err(404, "not_found", "Not found.")
}

impl StubRelay {
    pub(crate) fn pairing_route(&self, req: &HttpRequest, method: &str, path: &str, q: &HashMap<String, String>) -> Resp {
        let rest = path.trim_start_matches("/v1/pair").trim_start_matches('/');
        match (method, rest) {
            ("POST", "") => self.pair_create(req),
            (_, "") => err(405, "method_not_allowed", "Not allowed."),
            ("GET", pid) if !pid.contains('/') => self.pair_get(pid, q, req),
            ("POST", r) => match r.split_once('/') {
                Some((pid, "response")) => self.pair_response(pid, req),
                Some((pid, "decision")) => self.pair_decision(pid, req),
                Some((pid, "reject")) => self.pair_reject(pid, req, false),
                Some((pid, "burn")) => self.pair_reject(pid, req, true),
                _ => not_found(),
            },
            _ => not_found(),
        }
    }

    fn pair_create(&self, req: &HttpRequest) -> Resp {
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        if who.role != "desktop" {
            return err(403, "forbidden", "Only a desktop opens a rendezvous.");
        }
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let (Some(pid), Some(offer), Some(mac), Some(app), Some(thumb)) =
            (doc.get_str("pid"), doc.get_str("offer"), doc.get_str("mac"), doc.get_str("appId"), doc.get_str("desktopThumbprint"))
        else {
            return err(400, "invalid_request", "pid, offer, mac, appId and desktopThumbprint are required.");
        };
        if !ids::is_pid(pid)
            || offer.is_empty()
            || offer.len() > 4096
            || b64::decode_exact::<32>(mac).is_err()
            || !ids::is_app_id(app)
            || !ids::is_thumbprint(thumb)
        {
            return err(400, "invalid_request", "A field is not in its form.");
        }
        let ttl = match doc.get("ttl") {
            None => 600,
            Some(t) => match t.as_uint53() {
                Some(n) if (1..=900).contains(&n) => n as i64,
                _ => return err(400, "invalid_request", "ttl"),
            },
        };
        let Ok(offer_doc) = json::parse(offer.as_bytes()) else { return err(400, "invalid_request", "The offer is not JSON.") };
        let key = offer_doc.get("desktopEndpointKey");
        let (Some(offer_app), Some(offer_thumb), Some(conn)) =
            (offer_doc.get_str("appId"), key.and_then(|k| k.get_str("thumbprint")), offer_doc.get_str("desktopConnectionId"))
        else {
            return err(400, "invalid_request", "The offer lacks appId, desktopConnectionId or desktopEndpointKey.");
        };
        if offer_app != app || offer_thumb != thumb || conn != who.device {
            return err(400, "invalid_request", "The offer does not say what the request says.");
        }
        let Ok(desktop_ed) = VerifyKey::from_b64u(key.and_then(|k| k.get_str("publicKey")).unwrap_or("")) else {
            return err(422, "unprocessable", "A desktop key of small order.");
        };
        if desktop_ed.thumbprint() != thumb {
            return err(400, "invalid_request", "The thumbprint is not the key's.");
        }
        let now = self.now();
        if let Some(existing) = st.pairings.get(pid) {
            let same = existing.desktop == who.device
                && existing.offer == offer
                && existing.mac == mac
                && existing.app_id == app
                && existing.exp > now
                && existing.state != PState::Ended;
            if same {
                return Resp {
                    status: 201,
                    headers: Vec::new(),
                    body: Json::obj([("pid", Json::str(pid)), ("exp", Json::int(existing.exp)), ("time", Json::int(now))]).to_compact(),
                    time: None,
                };
            }
            if existing.exp > now && existing.state != PState::Ended {
                return err(409, "conflict", "This pid exists.");
            }
        }
        let exp = now + ttl;
        st.pairings.insert(
            pid.to_string(),
            Pairing {
                desktop: who.device,
                app_id: app.to_string(),
                offer: offer.to_string(),
                mac: mac.to_string(),
                desktop_ed,
                desktop_thumb: thumb.to_string(),
                exp,
                state: PState::Open,
                response: None,
                responses: 0,
                rejects: 0,
                phone_dev: None,
                sealed: None,
                receipt: None,
            },
        );
        Resp {
            status: 201,
            headers: Vec::new(),
            body: Json::obj([("pid", Json::str(pid)), ("exp", Json::int(exp)), ("time", Json::int(now))]).to_compact(),
            time: None,
        }
    }

    fn fetch_body(&self, p: &Pairing, now: i64) -> Json {
        let cfg = self.cfg();
        let mut m: Vec<(&str, Json)> = vec![("v", Json::int(1)), ("state", Json::str(state_word(p.state)))];
        match p.state {
            PState::Open | PState::Answered => {
                m.push(("offer", Json::str(p.offer.clone())));
                m.push(("mac", Json::str(p.mac.clone())));
                m.push(("exp", Json::int(p.exp)));
            }
            PState::Approved => {
                m.push(("deviceId", Json::str(p.phone_dev.clone().unwrap_or_default())));
                m.push(("sealedToken", Json::str(p.sealed.clone().unwrap_or_default())));
                let (issued, sig, grants) = p.receipt.clone().unwrap_or_default();
                let mut r = vec![("issuedAt", Json::int(issued)), ("signature", Json::str(sig))];
                if cfg.receipt_includes_grants {
                    r.push(("grants", Json::Arr(grants.into_iter().map(Json::str).collect())));
                }
                m.push(("receipt", Json::Obj(r.into_iter().map(|(k, v)| (k.to_string(), v)).collect())));
            }
            _ => {}
        }
        m.push(("time", Json::int(now)));
        Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    fn pair_get(&self, pid: &str, q: &HashMap<String, String>, req: &HttpRequest) -> Resp {
        let cfg = self.cfg();
        let mut st = self.lock();
        let seen = match q.get("state").map(String::as_str) {
            None => None,
            Some("open") => Some(PState::Open),
            Some("answered") => Some(PState::Answered),
            Some("approved") => Some(PState::Approved),
            Some("denied") => Some(PState::Denied),
            Some(_) => return err(400, "invalid_request", "state"),
        };
        let wait = match q.get("wait") {
            None => 0,
            Some(v) if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => v.parse::<u64>().unwrap_or(0).min(cfg.wait_max),
            Some(_) => return err(400, "invalid_request", "wait"),
        };
        let deadline = Instant::now() + Duration::from_secs(wait);
        // Counted in `GET /v1/admin/status` as a held request for as long as it waits.
        let mut counted = false;
        loop {
            let now = self.now();
            let live = ids::is_pid(pid) && st.pairings.get(pid).is_some_and(|p| p.exp > now && p.state != PState::Ended);
            if !live {
                if counted {
                    st.pair_waiting = st.pair_waiting.saturating_sub(1);
                }
                return not_found();
            }
            let p = &st.pairings[pid];
            let waitable = matches!(p.state, PState::Open | PState::Answered) && seen.is_none_or(|s| s == p.state);
            if wait > 0 && waitable && Instant::now() < deadline && !req.cancel.is_cancelled() {
                if !counted {
                    counted = true;
                    st.pair_waiting += 1;
                }
                let (g, _) = self.0.cv.wait_timeout(st, Duration::from_millis(40)).unwrap_or_else(|e| e.into_inner());
                st = g;
                continue;
            }
            let mut body = self.fetch_body(p, now);
            if counted {
                st.pair_waiting = st.pair_waiting.saturating_sub(1);
            }
            if wait > 0 {
                if let Json::Obj(m) = &mut body {
                    m.push(("hold".to_string(), Json::obj([("granted", Json::Bool(true))])));
                }
            }
            return ok(body);
        }
    }

    fn pair_response(&self, pid: &str, req: &HttpRequest) -> Resp {
        let mut st = self.lock();
        let now = self.now();
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let Some(text) = doc.get_str("response") else { return err(400, "invalid_request", "response is required.") };
        if text.is_empty() || text.len() > 8192 {
            return err(400, "invalid_request", "The response is 1 to 8192 bytes.");
        }
        match json::parse(text.as_bytes()) {
            Ok(r) if r.get_str("kind") == Some("aokie_mobile_pairing_response") => {}
            _ => return err(400, "invalid_request", "The response is not a pairing response."),
        }
        let live = ids::is_pid(pid) && st.pairings.get(pid).is_some_and(|p| p.exp > now && p.state != PState::Ended);
        if !live {
            return not_found();
        }
        let Some(p) = st.pairings.get_mut(pid) else { return not_found() };
        match p.state {
            PState::Open => {
                if p.responses >= 3 {
                    return err(409, "already_answered", "This rendezvous has had its three responses.");
                }
                p.responses += 1;
                p.response = Some(text.to_string());
                p.state = PState::Answered;
                let (desktop, n) = (p.desktop.clone(), p.responses);
                let item_id = if n == 1 { pid.to_string() } else { format!("{pid}.{n}") };
                let exp = p.exp;
                Self::enqueue(&mut st, &desktop, "pair", &item_id, "relay", Json::obj([("ct", Json::str("json"))]), text, now, exp.min(now + 900));
                self.0.cv.notify_all();
                Resp {
                    status: 202,
                    headers: Vec::new(),
                    body: Json::obj([("state", Json::str("answered")), ("time", Json::int(now))]).to_compact(),
                    time: None,
                }
            }
            _ if p.response.as_deref() == Some(text) => Resp {
                status: 202,
                headers: Vec::new(),
                body: Json::obj([("state", Json::str("answered")), ("time", Json::int(now))]).to_compact(),
                time: None,
            },
            _ => err(409, "already_answered", "This offer was already answered."),
        }
    }

    fn pair_reject(&self, pid: &str, req: &HttpRequest, burn: bool) -> Resp {
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let now = self.now();
        let Some(p) = st.pairings.get_mut(pid).filter(|p| p.desktop == who.device) else { return not_found() };
        if p.exp <= now {
            return err(410, "expired", "Expired.");
        }
        if p.state == PState::Ended {
            return if burn {
                ok(Json::obj([("v", Json::int(1)), ("state", Json::str("expired")), ("time", Json::int(now))]))
            } else {
                err(410, "expired", "Expired.")
            };
        }
        let answer = if burn {
            if p.state == PState::Approved {
                return err(409, "conflict", "The phone has yet to read its token.");
            }
            p.state = PState::Ended;
            "expired"
        } else {
            if p.state != PState::Answered {
                return err(409, "conflict", "Nothing to reject.");
            }
            p.response = None;
            p.rejects += 1;
            if p.rejects >= 3 {
                p.state = PState::Ended;
                "expired"
            } else {
                p.state = PState::Open;
                "open"
            }
        };
        self.0.cv.notify_all();
        ok(Json::obj([("v", Json::int(1)), ("state", Json::str(answer)), ("time", Json::int(now))]))
    }

    fn pair_decision(&self, pid: &str, req: &HttpRequest) -> Resp {
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let now = self.now();
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let Some(approve) = doc.get("approve").and_then(Json::as_bool) else { return err(400, "invalid_request", "approve is required.") };
        let Some(p) = st.pairings.get(pid).filter(|p| p.desktop == who.device) else { return not_found() };
        if p.exp <= now || p.state == PState::Ended {
            return err(410, "expired", "Expired.");
        }
        if !approve {
            if p.state != PState::Answered && p.state != PState::Denied {
                return err(409, "conflict", "Nothing to deny.");
            }
            if let Some(p) = st.pairings.get_mut(pid) {
                p.state = PState::Denied;
            }
            self.0.cv.notify_all();
            return ok(Json::obj([("v", Json::int(1)), ("state", Json::str("denied")), ("time", Json::int(now))]));
        }
        if p.state == PState::Approved {
            // The same approval again (the desktop's outbox retries) answers the same; one for another key is a conflict (Pairing::decide of the relay).
            let thumb = doc.get("phone").and_then(|p| p.get_str("thumbprint")).unwrap_or("");
            let device = p.phone_dev.clone().unwrap_or_default();
            let same = !thumb.is_empty() && st.devices.iter().find(|d| d.id == device).and_then(|d| d.thumbprint.as_deref()) == Some(thumb);
            if !same {
                return err(409, "conflict", "This rendezvous was approved for another phone.");
            }
            return ok(Json::obj([("v", Json::int(1)), ("state", Json::str("approved")), ("deviceId", Json::str(device)), ("time", Json::int(now))]));
        }
        if p.state != PState::Answered {
            return err(409, "conflict", "There is no response to approve.");
        }
        // What the relay checks of an approval (Interpretation 31): shape is 400, a key or a mismatch is 422, and the receipt is the desktop's own signature.
        let (Some(phone), Some(name), Some(app), Some(grants), Some(receipt)) =
            (doc.get("phone"), doc.get_str("name"), doc.get_str("appId"), doc.get("grants").and_then(Json::as_array), doc.get("receipt"))
        else {
            return err(400, "invalid_request", "phone, name, appId, grants and receipt are required.");
        };
        let (Some(ed_text), Some(x_text), Some(thumb)) = (phone.get_str("ed25519"), phone.get_str("x25519"), phone.get_str("thumbprint")) else {
            return err(400, "invalid_request", "phone keys");
        };
        let (Some(issued), Some(sig)) = (receipt.get_uint53("issuedAt"), receipt.get_str("signature")) else {
            return err(400, "invalid_request", "receipt");
        };
        let grants: Vec<String> = grants.iter().filter_map(|g| g.as_str().map(str::to_string)).collect();
        if grants.len() != doc.get("grants").and_then(Json::as_array).map_or(0, <[Json]>::len)
            || grants.len() > 16
            || grants.iter().any(|g| !ids::is_grant(g))
            || ids::clean_name(name, 60).is_empty()
        {
            return err(400, "invalid_request", "grants or name");
        }
        let (Ok(ed), Ok(x)) = (VerifyKey::from_b64u(ed_text), X25519Public::from_b64u(x_text)) else {
            return err(422, "unprocessable", "A phone key of small order.");
        };
        if app != p.app_id || ed.thumbprint() != thumb || grants.iter().any(|g| !ids::is_known_grant(g)) {
            return err(422, "unprocessable", "The approval does not fit the rendezvous.");
        }
        let claims = p.response.as_deref().and_then(|r| json::parse(r.as_bytes()).ok()).and_then(|r| r.get("claims").cloned());
        let same = claims.as_ref().is_some_and(|c| {
            c.get("mobileEndpointKey").and_then(|k| k.get_str("thumbprint")) == Some(thumb)
                && c.get("mobileEndpointKey").and_then(|k| k.get_str("publicKey")) == Some(ed_text)
                && c.get_str("mobileX25519") == Some(x_text)
        });
        if !same {
            return err(422, "unprocessable", "The keys approved are not the keys the phone answered with.");
        }
        let receipt_ok = math::verify_receipt(&p.desktop_ed, &p.app_id, &grants, issued, thumb, pid, sig).is_ok();
        if !receipt_ok || thumbprint_of(&p.desktop_ed.to_bytes()) != p.desktop_thumb {
            return err(422, "unprocessable", "The receipt does not verify.");
        }
        let owner = p.desktop.clone();
        let grant_refs: Vec<&str> = grants.iter().map(String::as_str).collect();
        let (device, token) =
            self.create_device_locked(&mut st, "phone", &ids::clean_name(name, 60), Some(&owner), Some(app), &grant_refs, Some(&ed));
        let desktop_thumb = p_desktop_thumb(&st, pid);
        if let Some(d) = st.devices.iter_mut().find(|d| d.id == device) {
            d.peer_thumbprint = Some(desktop_thumb);
        }
        let Ok(sealed) = x.seal(token.as_bytes()) else { return err(422, "unprocessable", "The token could not be sealed.") };
        if let Some(p) = st.pairings.get_mut(pid) {
            p.state = PState::Approved;
            p.phone_dev = Some(device.clone());
            p.sealed = Some(b64::encode(&sealed));
            p.receipt = Some((issued, sig.to_string(), grants));
            p.response = None;
        }
        self.0.cv.notify_all();
        ok(Json::obj([("v", Json::int(1)), ("state", Json::str("approved")), ("deviceId", Json::str(device)), ("time", Json::int(now))]))
    }
}

fn state_word(s: PState) -> &'static str {
    match s {
        PState::Open => "open",
        PState::Answered => "answered",
        PState::Approved => "approved",
        PState::Denied => "denied",
        PState::Ended => "expired",
    }
}

fn p_desktop_thumb(st: &super::stub::State, pid: &str) -> String {
    st.pairings.get(pid).map(|p| p.desktop_thumb.clone()).unwrap_or_default()
}
