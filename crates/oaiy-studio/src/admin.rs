//! The control port: the UI page, its JSON API, a playground that reaches every
//! gateway target without a key, and the generated files.
//!
//! This port can change which programs the studio runs, so it answers only
//! requests addressed to it by name (a `Host` of this listener: DNS rebinding
//! cannot reach it) and, for anything but plain page loads, only from its own
//! origin (another web page open in the same browser cannot drive it).

use crate::gateway::{self, json_reply, Matched};
use crate::util::{error_json, str_or};
use crate::{config, registry, Studio};
use oaiy_engine::http::{respond, Request};
use oaiy_engine::json::Json;
use std::io;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;

const PAGE: &str = include_str!("ui.html");
/// The page's type, OAIY's (SIL OFL 1.1; the licences are beside them in `fonts/`).
const FONTS: [(&str, &[u8]); 2] = [
    ("/fonts/public-sans.woff2", include_bytes!("fonts/public-sans.woff2")),
    ("/fonts/jetbrains-mono.woff2", include_bytes!("fonts/jetbrains-mono.woff2")),
];

fn err(w: &mut TcpStream, status: u16, message: &str) -> io::Result<bool> {
    json_reply(w, status, &error_json(message, "invalid_request_error", "invalid_request"))
}

fn body(req: &Request) -> Result<Json, String> {
    Json::parse(&req.body).map_err(|e| format!("request body: {e}"))
}

/// Hosts this listener answers to: its own address, `localhost` and loopback
/// on its port, and the configured host.
pub fn allowed_host(host: &str, port: u16, configured: &str) -> bool {
    let (name, p) = match host.rsplit_once(':') {
        Some((n, p)) if !n.ends_with(']') || n.starts_with('[') => (n, p.parse::<u16>().ok()),
        _ => (host, Some(80)),
    };
    if p != Some(port) {
        return false;
    }
    // Listening on every interface: clients arrive by whatever LAN name or
    // address they use. The API key (required for non-loopback) guards it then.
    if configured == "0.0.0.0" || configured == "::" {
        return true;
    }
    let name = name.trim_start_matches('[').trim_end_matches(']');
    ["localhost", "127.0.0.1", "::1", configured].iter().any(|h| h.eq_ignore_ascii_case(name))
}

pub fn handle(studio: &Arc<Studio>, req: &Request, w: &mut TcpStream, port: u16) -> io::Result<bool> {
    let cfg = studio.config();
    let ui_host = cfg.get("ui").map_or("127.0.0.1", |u| str_or(u, "host", "127.0.0.1")).to_string();
    if !req.header("host").is_some_and(|h| allowed_host(h, port, &ui_host)) {
        return err(w, 403, "unexpected Host header");
    }
    if let Some(origin) = req.header("origin") {
        let own = req.header("host").map(|h| format!("http://{h}"));
        if own.as_deref() != Some(origin) {
            return err(w, 403, "cross-origin requests are not accepted on the control port");
        }
    }
    // Reached from another machine: the gateway key guards the controls too.
    // This machine's own clients (the app window, a second launch looking for
    // this one) connect over loopback and are let through, as on a local bind.
    let exposed = !(ui_host == "localhost" || ui_host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback()));
    let remote = exposed && !w.peer_addr().is_ok_and(|a| a.ip().is_loopback() || a.ip().to_canonical().is_loopback());
    let key = cfg.get("gateway").map_or("", |g| str_or(g, "api_key", "")).to_string();
    let route = req.route().to_string();
    let font = FONTS.iter().find(|(path, _)| *path == route).map(|(_, bytes)| *bytes);
    if remote && route != "/" && font.is_none() && !key.is_empty() {
        let given = req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")).map(str::trim).map(str::to_string).or_else(|| req.query("key"));
        if given.as_deref() != Some(key.as_str()) {
            return err(w, 401, "this control port needs the gateway API key");
        }
    }
    let method = req.method.as_str();
    if method == "OPTIONS" {
        respond(w, 204, "text/plain", b"", true)?;
        return Ok(true);
    }
    if method == "GET" && (route == "/" || route == "/index.html") {
        respond(w, 200, "text/html; charset=utf-8", PAGE.as_bytes(), true)?;
        return Ok(true);
    }
    if let (true, Some(bytes)) = (method == "GET", font) {
        oaiy_engine::http::respond_with(w, 200, "font/woff2", &[("Cache-Control", "max-age=86400")], bytes, true)?;
        return Ok(true);
    }
    if let Some(rel) = route.strip_prefix("/files/") {
        return gateway::files(studio, req, w, rel);
    }
    if let Some(rest) = route.strip_prefix("/api/play/") {
        // The playground: every target, no key, local references allowed.
        let (target, sub) = rest.split_once('/').map_or((rest, String::new()), |(t, s)| (t, format!("/{s}")));
        let target = match target {
            "chat" | "completions" | "models" | "images" | "edits" | "videos" | "speech" | "voices" | "music" | "sound" | "model3d" | "health" => target,
            _ => return err(w, 404, "unknown playground target"),
        };
        let m = Matched { target: target.into(), spec: "openai".into(), rest: sub };
        return gateway::handle(studio, req, w, m, true);
    }
    if method == "GET" && route == "/api/models/export" {
        let file = registry::export(&cfg, &studio.root);
        let body = config::pretty(&file, 0);
        oaiy_engine::http::respond_with(w, 200, "application/json", &[("Content-Disposition", "attachment; filename=\"oaiy-models.json\"")], body.as_bytes(), true)?;
        return Ok(true);
    }
    let result: Result<Json, (u16, String)> = match (method, route.as_str()) {
        ("GET", "/api/state") => Ok(studio.state()),
        ("GET", "/api/discovery") => {
            let base = studio.state().get("gateway_url").and_then(Json::as_str).unwrap_or("").to_string();
            Ok(crate::discovery::document(studio, &base, true))
        }
        ("GET", "/api/config") => Ok(cfg.clone()),
        ("PUT" | "POST", "/api/config") => body(req).map_err(|e| (400, e)).and_then(|v| studio.set_config(v).map_err(|e| (400, e))),
        ("POST", "/api/config/reset-routes") => {
            let mut next = cfg.clone();
            let defaults = config::default_json();
            if let (Some(g), Some(routes)) = (next.get("gateway").cloned(), defaults.get("gateway").and_then(|g| g.get("routes")).cloned()) {
                let mut g = g;
                crate::util::set(&mut g, "routes", routes);
                crate::util::set(&mut next, "gateway", g);
            }
            studio.set_config(next).map_err(|e| (400, e))
        }
        ("GET", "/api/jobs") => {
            let root = studio.output_root();
            Ok(Json::obj([("jobs", Json::Arr(studio.media.list().iter().map(|j| j.to_json(&root)).collect()))]))
        }
        ("GET", "/api/system") => Ok(Json::obj([("gpus", studio.system.gpus()), ("ram", studio.system.ram())])),
        ("GET", "/api/logs") => {
            let after = req.query("after").and_then(|a| a.parse().ok()).unwrap_or(0);
            let ring = match req.query("source").as_deref() {
                Some("llm") => &studio.llm.log,
                Some("media") => &studio.media.log,
                _ => &studio.log,
            };
            Ok(Json::obj([("lines", ring.since(after))]))
        }
        ("POST", "/api/llm/start") => studio.llm.start(&cfg, &studio.root).map(|_| studio.llm.status()).map_err(|e| (400, e)),
        ("POST", "/api/llm/stop") => {
            studio.llm.stop();
            Ok(studio.llm.status())
        }
        ("POST", "/api/llm/restart") => {
            studio.llm.stop();
            studio.llm.start(&cfg, &studio.root).map(|_| studio.llm.status()).map_err(|e| (400, e))
        }
        ("GET", "/api/downloads") => Ok(studio.downloads.state(studio)),
        ("POST", "/api/downloads") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            studio.downloads.start(studio, str_or(&b, "id", ""), dir).map_err(|e| (400, e))
        }),
        ("POST", "/api/downloads/pause") => body(req).map_err(|e| (400, e)).map(|b| studio.downloads.pause(studio, str_or(&b, "id", ""))),
        ("POST", "/api/downloads/cancel") => body(req).map_err(|e| (400, e)).map(|b| studio.downloads.cancel(studio, str_or(&b, "id", ""))),
        ("POST", "/api/downloads/add") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            studio.downloads.add_again(studio, str_or(&b, "id", ""), dir).map_err(|e| (400, e))
        }),
        // Where downloads go, and the Hugging Face token for gated models (never sent back).
        ("POST", "/api/downloads/settings") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let mut next = cfg.clone();
            let mut d = next.get("downloads").cloned().unwrap_or(Json::obj(Vec::<(&str, Json)>::new()));
            if let Some(dir) = b.get("dir").and_then(Json::as_str) {
                crate::util::set(&mut d, "dir", Json::str(dir.trim()));
            }
            if let Some(t) = b.get("hf_token").and_then(Json::as_str) {
                crate::util::set(&mut d, "hf_token", Json::str(t.trim()));
            }
            crate::util::set(&mut next, "downloads", d);
            studio.set_config(next).map_err(|e| (400, e))?;
            Ok(studio.downloads.state(studio))
        }),
        ("GET", "/api/browse") => crate::system::browse(req.query("path").as_deref()).map_err(|e| (400, e)),
        ("POST", "/api/detect") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let path = PathBuf::from(str_or(&b, "path", ""));
            crate::detect::detect(&path).map(|d| d.to_json()).map_err(|e| (400, e))
        }),
        ("POST", "/api/models/add") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let path = PathBuf::from(str_or(&b, "path", "").trim());
            let target = match (b.get("target_section").and_then(Json::as_str), b.get("target_model").and_then(Json::as_str)) {
                (Some(s), Some(m)) => Some((s.to_string(), m.to_string())),
                _ => None,
            };
            let mut next = cfg.clone();
            let (added, detected) = registry::add(&mut next, &path, b.get("name").and_then(Json::as_str), target.as_ref().map(|(s, m)| (s.as_str(), m.as_str())))
                .map_err(|e| (400, e))?;
            studio.set_config(next).map_err(|e| (400, e))?;
            studio.log.push(format!("added {} ({}) as {} model {}", path.display(), detected.summary, added.section, added.name));
            Ok(Json::obj([
                ("section", Json::str(added.section)),
                ("name", Json::str(&added.name)),
                ("enabled", Json::Bool(added.enabled)),
                ("missing", Json::Arr(added.missing.iter().map(Json::str).collect())),
                ("detected", detected.to_json()),
            ]))
        }),
        ("POST", "/api/models/import") => body(req).map_err(|e| (400, e)).and_then(|doc| {
            let replace = req.query("mode").as_deref() == Some("replace");
            let mut next = cfg.clone();
            let report = registry::import(&mut next, &doc, replace, &studio.root).map_err(|e| (400, e))?;
            studio.set_config(next).map_err(|e| (400, e))?;
            studio.log.push(format!("imported models ({})", if replace { "replaced" } else { "merged" }));
            Ok(report)
        }),
        ("POST", "/api/models/remove") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let mut next = cfg.clone();
            remove_model(&mut next, str_or(&b, "section", ""), str_or(&b, "name", "")).map_err(|e| (400, e))?;
            studio.set_config(next).map_err(|e| (400, e))
        }),
        ("POST", r) if r.strip_prefix("/api/jobs/").and_then(|x| x.strip_suffix("/cancel")).is_some_and(|id| !id.is_empty()) => {
            let id = r.strip_prefix("/api/jobs/").and_then(|x| x.strip_suffix("/cancel")).unwrap_or_default();
            Ok(Json::obj([("cancelled", Json::Bool(studio.media.cancel(id)))]))
        }
        ("DELETE", r) if r.strip_prefix("/api/jobs/").is_some_and(|id| !id.is_empty() && !id.contains('/')) => {
            let id = r.strip_prefix("/api/jobs/").unwrap_or_default();
            Ok(Json::obj([("removed", Json::Bool(studio.media.remove(id)))]))
        }
        ("POST", "/api/music/quantize") => body(req).map_err(|e| (400, e)).and_then(|b| {
            // Make a music model's smaller language model; when it is written,
            // the model uses it.
            let name = str_or(&b, "model", "").to_string();
            let quant = str_or(&b, "quant", "q4_k").to_string();
            let (request, out) = crate::music::quantize_request(&cfg, &studio.root, &name, &quant).map_err(|e| (400, e))?;
            let job = studio.media.submit(crate::media::Kind::Music, request, name.clone(), format!("smaller copy ({quant})"), 1, 0.0, 200, None);
            let s = Arc::clone(studio);
            let id = job.id.clone();
            let reply = Json::obj([("job", Json::str(&job.id)), ("output", Json::str(out.to_string_lossy()))]);
            std::thread::spawn(move || {
                let Some(done) = s.media.wait(&id, std::time::Duration::from_secs(4 * 3600)) else { return };
                if done.status != "completed" {
                    return;
                }
                let mut next = s.config();
                let set_it = crate::registry::obj_mut(&mut next, &["media", "music", "models", &name]).map(|m| {
                    crate::util::set(m, "language_model", crate::detect::path_json(&out));
                });
                if set_it.is_some() {
                    match s.set_config(next) {
                        Ok(_) => s.log.push(format!("music model {name} now uses {}", out.display())),
                        Err(e) => s.log.push(format!("music model {name}: could not use {}: {e}", out.display())),
                    }
                }
            });
            Ok(reply)
        }),
        ("POST", "/api/open-outputs") => {
            let dir = studio.output_root();
            let _ = std::fs::create_dir_all(&dir);
            crate::open_path(&dir.to_string_lossy());
            Ok(Json::obj([("opened", Json::str(dir.to_string_lossy()))]))
        }
        _ => Err((404, format!("no route {method} {route}"))),
    };
    match result {
        Ok(v) => json_reply(w, 200, &v),
        Err((status, message)) => err(w, status, &message),
    }
}

fn remove_model(cfg: &mut Json, section: &str, name: &str) -> Result<(), String> {
    let clear_default = |sec: &mut Json| {
        if str_or(sec, "default_model", "") == name {
            crate::util::set(sec, "default_model", Json::str(""));
        }
    };
    let Json::Obj(top) = cfg else { return Err("bad configuration".into()) };
    match section {
        "llm" => {
            let llm = &mut top.iter_mut().find(|(k, _)| k == "llm").ok_or("no llm")?.1;
            if let Json::Obj(fields) = llm {
                if let Some((_, Json::Arr(models))) = fields.iter_mut().find(|(k, _)| k == "models") {
                    let before = models.len();
                    models.retain(|m| str_or(m, "name", "") != name);
                    if models.len() == before {
                        return Err(format!("no llm model {name}"));
                    }
                }
            }
            clear_default(llm);
        }
        "image" | "video" | "speech" | "music" | "sound" | "model3d" => {
            let media = &mut top.iter_mut().find(|(k, _)| k == "media").ok_or("no media")?.1;
            let Json::Obj(media) = media else { return Err("bad media".into()) };
            let sec = &mut media.iter_mut().find(|(k, _)| k == section).ok_or("no section")?.1;
            if let Json::Obj(fields) = sec {
                if let Some((_, Json::Obj(models))) = fields.iter_mut().find(|(k, _)| k == "models") {
                    let before = models.len();
                    models.retain(|(k, _)| k != name);
                    if models.len() == before {
                        return Err(format!("no {section} model {name}"));
                    }
                }
            }
            clear_default(sec);
        }
        _ => return Err("section must be llm, image, video, speech, music, sound or model3d".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_has_oaiy_s_type() {
        for (path, bytes) in FONTS {
            assert!(PAGE.contains(path), "{path} is not used by the page");
            assert_eq!(&bytes[..4], b"wOF2", "{path}");
        }
    }

    /// The page's sections and its menu are one list, in one order: every
    /// section is in the menu under a name (the menu shows names at every
    /// width, never pictures alone), and every entry opens a section that is there.
    #[test]
    fn every_section_of_the_page_is_in_its_menu_by_name() {
        let start = PAGE.find(r#"<nav class="sections""#).expect("the page has its menu");
        let menu = &PAGE[start..start + PAGE[start..].find("</nav>").expect("the menu ends")];
        let quoted = |text: &str| text[..text.find('"').expect("a closing quote")].to_string();
        let in_menu: Vec<String> = menu.match_indices(r#"data-view=""#).map(|(at, mark)| quoted(&menu[at + mark.len()..])).collect();
        let pages: Vec<String> = PAGE
            .match_indices(r#"<section class="view"#)
            .map(|(at, _)| {
                let rest = &PAGE[at..];
                let id = r#"id="v-"#;
                quoted(&rest[rest.find(id).expect("a section has its id") + id.len()..])
            })
            .collect();
        assert!(pages.len() >= 9, "{pages:?}");
        assert_eq!(in_menu, pages, "the menu's entries and the page's sections");
        assert_eq!(menu.matches(r#"<span class="label">"#).count(), in_menu.len() + 1, "each entry has its name, and More has");
        // The four a narrow window keeps in the bar, and the rest under More: none is in neither.
        assert_eq!(menu.matches(" data-more>").count(), in_menu.len() - 4, "the entries More holds");
    }

    #[test]
    fn only_this_listener_s_names_are_answered() {
        assert!(allowed_host("127.0.0.1:7860", 7860, "127.0.0.1"));
        assert!(allowed_host("localhost:7860", 7860, "127.0.0.1"));
        assert!(allowed_host("[::1]:7860", 7860, "127.0.0.1"));
        assert!(allowed_host("studio.lan:7860", 7860, "studio.lan"));
        assert!(!allowed_host("evil.example:7860", 7860, "127.0.0.1"));
        assert!(!allowed_host("127.0.0.1:8080", 7860, "127.0.0.1"));
        assert!(!allowed_host("localhost", 7860, "127.0.0.1"));
        assert!(allowed_host("192.168.1.20:7860", 7860, "0.0.0.0"));
        assert!(!allowed_host("192.168.1.20:9999", 7860, "0.0.0.0"));
    }

    #[test]
    fn removing_a_model_clears_it_as_the_default() {
        let mut cfg = config::default_json();
        let mut llm = cfg.get("llm").unwrap().clone();
        crate::util::set(&mut llm, "models", Json::parse(br#"[{"name":"a","path":"a.gguf"}]"#).unwrap());
        crate::util::set(&mut llm, "default_model", Json::str("a"));
        crate::util::set(&mut cfg, "llm", llm);
        remove_model(&mut cfg, "llm", "a").unwrap();
        assert_eq!(cfg.get("llm").unwrap().get("default_model").and_then(Json::as_str), Some(""));
        assert!(remove_model(&mut cfg, "llm", "a").is_err());
        config::validate(&cfg).unwrap();
        // Sound effect and 3D models are removed the same way.
        for section in ["sound", "model3d"] {
            let media = crate::registry::obj_mut(&mut cfg, &["media", section]).unwrap();
            crate::util::set(media, "models", Json::parse(br#"{"m":{"path":"m"}}"#).unwrap());
            crate::util::set(media, "default_model", Json::str("m"));
            remove_model(&mut cfg, section, "m").unwrap();
            assert_eq!(cfg.get("media").unwrap().get(section).unwrap().get("default_model").and_then(Json::as_str), Some(""));
        }
        config::validate(&cfg).unwrap();
    }
}
