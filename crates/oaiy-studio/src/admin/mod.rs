//! The control port: the UI page, its JSON API, a playground that reaches every
//! gateway target without a key, and the generated files.
//!
//! This port can change which programs the studio runs, so it answers only
//! requests addressed to it by name (a `Host` of this listener: DNS rebinding
//! cannot reach it) and, for anything but plain page loads, only from its own
//! origin (another web page open in the same browser cannot drive it).

use crate::api::gateway::{self, json_reply, Matched};
use crate::util::{error_json, str_or};
use crate::{config, models::registry, Studio};
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
            Ok(crate::api::discovery::document(studio, &base, true))
        }
        // (the settings with their version, which a save of them sends back, and their secrets said to be set, not said)
        ("GET", "/api/config") => Ok(studio.config_for_page()),
        ("PUT" | "POST", "/api/config") => body(req).map_err(|e| (400, e)).and_then(|v| studio.save_config_from(v)).map(|_| studio.config_for_page()),
        ("POST", "/api/config/reset-routes") => studio
            .update_config(|next| {
                let defaults = config::default_json();
                if let (Some(g), Some(routes)) = (next.get("gateway").cloned(), defaults.get("gateway").and_then(|g| g.get("routes")).cloned()) {
                    let mut g = g;
                    crate::util::set(&mut g, "routes", routes);
                    crate::util::set(next, "gateway", g);
                }
                Ok(())
            })
            .map(|_| studio.config_for_page()),
        ("GET", "/api/jobs") => {
            let root = studio.output_root();
            Ok(Json::obj([("jobs", Json::Arr(studio.media.list().iter().map(|j| j.to_json(&root)).collect()))]))
        }
        ("GET", "/api/system") => Ok(Json::obj([("gpus", studio.system.gpus()), ("ram", studio.system.ram())])),
        ("GET", "/api/logs") => {
            let after = req.query("after").and_then(|a| a.parse().ok()).unwrap_or(0);
            let ring = match req.query("source").as_deref() {
                Some("llm") => &studio.llm.log,
                Some("egpu") => &studio.egpu.log,
                Some("media") => &studio.media.log,
                _ => &studio.log,
            };
            Ok(Json::obj([("lines", ring.since(after))]))
        }
        // The page's "Language model" card: this computer's engine's, and the card's where every model is the card's
        // ([`llm_start`]).
        ("POST", "/api/llm/start") => llm_start(studio, &cfg).map(|_| studio.llm.status()).map_err(|e| (400, e)),
        ("POST", "/api/llm/stop") => {
            llm_stop(studio, &cfg);
            Ok(studio.llm.status())
        }
        ("POST", "/api/llm/restart") => {
            llm_stop(studio, &cfg);
            llm_start(studio, &cfg).map(|_| studio.llm.status()).map_err(|e| (400, e))
        }
        // tinygrad's server on a Mac's eGPU: what the Pythons here have, and its start (with the model named, else
        // the first set to it) and stop by hand. It starts by itself on the first request for a model of its own.
        ("GET", "/api/egpu/check") => Ok(studio.egpu.check(&cfg, &studio.root)),
        ("POST", "/api/egpu/start") => {
            let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
            let asked = body(req).ok().and_then(|b| b.get("model").and_then(Json::as_str).map(str::to_owned));
            egpu_start(studio, &cfg, asked).map(|_| studio.egpu.status(&llm)).map_err(|e| (400, e))
        }
        ("POST", "/api/egpu/stop") => {
            studio.egpu.stop();
            Ok(studio.egpu.status(cfg.get("llm").unwrap_or(&Json::Null)))
        }
        // What is loaded on each GPU, what is using it, and what loaded and let it go (`crate::resources`); memory
        // freed by hand: the picture worker and the lent card (`pictures`), the language models unless pinned
        // (`language`), or both (`all`).
        ("GET", "/api/resources") => Ok(crate::resources::snapshot(studio)),
        ("POST", "/api/resources/release") => (|| -> Result<Json, (u16, String)> {
            let what = body(req).ok().map(|b| str_or(&b, "what", "pictures").to_string()).unwrap_or_else(|| "pictures".into());
            if !matches!(what.as_str(), "pictures" | "language" | "all") {
                return Err((400, "what must be pictures, language or all".into()));
            }
            if what != "language" {
                studio.media.release_now();
            }
            if what != "pictures" {
                if crate::media::pinned(&cfg) {
                    return Err((409, "the language model is pinned (llm.pinned): unpin it to let it go".into()));
                }
                crate::resources::event("release", "", "", "asked: the language models let go");
                studio.llm.stop();
                // (the card's own server is the one a lent card's picture worker uses: not while lent)
                if !studio.egpu.is_lent() {
                    studio.egpu.stop();
                }
            }
            Ok(crate::resources::snapshot(studio))
        })(),
        // A model kept loaded (`crate::media::pins`): a picture model pinned in the picture worker on its GPU, or the
        // language model chosen in Engines (`llm.pinned`); unpinned, it is let go as any other again. Saved in the
        // settings, so a pin outlives a restart; refused where the pins there are cannot be kept beside it.
        ("POST", "/api/resources/pins") | ("POST", "/api/resources/unpin") => (|| -> Result<Json, (u16, String)> {
            let pin = route == "/api/resources/pins";
            let b = body(req).map_err(|e| (400, e))?;
            let model = str_or(&b, "model", "").trim().to_string();
            if model.is_empty() || model.len() > 200 || model.contains(['/', '\\']) || model.contains("..") {
                return Err((400, "model must be a model's name, as /v1/discovery lists it".into()));
            }
            // (each pin or unpin on the settings as they are now, under the save's lock: two at once, or a pin beside
            // another save, would each keep its own change and lose the other's)
            let kind = str_or(&b, "kind", "image").to_string();
            let changed = studio.update_config(|next| {
                if kind == "llm" {
                    let mut section = next.get("llm").cloned().unwrap_or(Json::Null);
                    if crate::llm::default_model(&section) != model {
                        return Err((409, format!("{model} is not the language model chosen in Engines: choose it first, then pin it")));
                    }
                    crate::util::set(&mut section, "pinned", Json::Bool(pin));
                    crate::util::set(next, "llm", section);
                    return Ok(true);
                }
                if pin {
                    crate::media::pins::check(next, &model).map_err(|e| (409, e))?;
                } else if !crate::media::pins::pinned(next).contains(&model) {
                    return Ok(false);
                }
                *next = crate::media::pins::with(next, &model, pin);
                Ok(true)
            })?;
            if !changed.0 {
                return Ok(crate::resources::snapshot(studio));
            }
            if !pin {
                // (a loading it was waiting for is not wanted any more; the job id read, then the pins let go, before
                // the queue is locked to cancel it)
                let loading = match studio.media.pin_states().get(&model) {
                    Some(crate::media::pins::State::Warming(job)) => Some(job.clone()),
                    _ => None,
                };
                if let Some(job) = loading {
                    studio.media.cancel(&job);
                }
            }
            crate::resources::event(if pin { "pin" } else { "unpin" }, &model, "", if pin { "pinned: kept loaded" } else { "unpinned: let go as any other model again" });
            Ok(crate::resources::snapshot(studio))
        })(),
        ("GET", "/api/downloads") => Ok(studio.downloads.state(studio)),
        // (`hf_token`: one for this download alone, the desktop's from its Settings; never kept or said back)
        ("POST", "/api/downloads") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            let token = Some(str_or(&b, "hf_token", "").trim().to_string()).filter(|t| !t.is_empty());
            studio.downloads.start(studio, str_or(&b, "id", ""), dir, token).map_err(|e| (400, e))
        }),
        // Where this computer keeps model files, by their SHA-256 (for another OAIY computer on the network that wants
        // one: its desktop serves the file, never its path).
        ("POST", "/api/library/find") => body(req).map_err(|e| (400, e)).map(|b| {
            let files: Vec<(String, u64)> = b
                .get("files")
                .and_then(Json::as_array)
                .unwrap_or(&[])
                .iter()
                .take(256)
                .filter_map(|f| {
                    let sha = str_or(f, "sha256", "").to_ascii_lowercase();
                    let size = f.get("size").and_then(Json::as_i64).filter(|s| *s > 0)? as u64;
                    (sha.len() == 64 && sha.bytes().all(|c| c.is_ascii_hexdigit())).then_some((sha, size))
                })
                .collect();
            studio.downloads.find_files(studio, &files, b.get("check").and_then(Json::as_bool).unwrap_or(false))
        }),
        // What getting a model takes: its files here already, on this computer elsewhere, or to download.
        // (`dir`: the folder it would go to, as the page has it, saved or not)
        ("POST", "/api/downloads/plan") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let token = Some(str_or(&b, "hf_token", "").trim().to_string()).filter(|t| !t.is_empty());
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            studio.downloads.plan_of(studio, str_or(&b, "id", ""), token, dir).map_err(|e| (400, e))
        }),
        ("POST", "/api/downloads/pause") => body(req).map_err(|e| (400, e)).map(|b| studio.downloads.pause(studio, str_or(&b, "id", ""))),
        // (`dir`: where a model is partly downloaded that no listed download says, after a restart)
        ("POST", "/api/downloads/cancel") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            studio.downloads.cancel(studio, str_or(&b, "id", ""), dir).map_err(|e| (409, e))
        }),
        ("POST", "/api/downloads/add") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let dir = Some(str_or(&b, "dir", "").trim()).filter(|d| !d.is_empty()).map(PathBuf::from);
            studio.downloads.add_again(studio, str_or(&b, "id", ""), dir).map_err(|e| (400, e))
        }),
        // Where downloads go, and the Hugging Face token for gated models (never sent back).
        ("POST", "/api/downloads/settings") => body(req).map_err(|e| (400, e)).and_then(|b| {
            studio.update_config(|next| {
                let mut d = next.get("downloads").cloned().unwrap_or(Json::obj(Vec::<(&str, Json)>::new()));
                if let Some(dir) = b.get("dir").and_then(Json::as_str) {
                    crate::util::set(&mut d, "dir", Json::str(dir.trim()));
                }
                if let Some(t) = b.get("hf_token").and_then(Json::as_str) {
                    crate::util::set(&mut d, "hf_token", Json::str(t.trim()));
                }
                crate::util::set(next, "downloads", d);
                Ok(())
            })?;
            Ok(studio.downloads.state(studio))
        }),
        ("GET", "/api/browse") => crate::system::browse(req.query("path").as_deref()).map_err(|e| (400, e)),
        ("POST", "/api/detect") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let path = PathBuf::from(str_or(&b, "path", ""));
            crate::models::detect::detect(&path).map(|d| d.to_json()).map_err(|e| (400, e))
        }),
        ("POST", "/api/models/add") => body(req).map_err(|e| (400, e)).and_then(|b| {
            let path = PathBuf::from(str_or(&b, "path", "").trim());
            let target = match (b.get("target_section").and_then(Json::as_str), b.get("target_model").and_then(Json::as_str)) {
                (Some(s), Some(m)) => Some((s.to_string(), m.to_string())),
                _ => None,
            };
            let ((added, detected), _) = studio.update_config(|next| {
                registry::add(next, &path, b.get("name").and_then(Json::as_str), target.as_ref().map(|(s, m)| (s.as_str(), m.as_str()))).map_err(|e| (400, e))
            })?;
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
            let (report, _) = studio.update_config(|next| registry::import(next, &doc, replace, &studio.root).map_err(|e| (400, e)))?;
            studio.log.push(format!("imported models ({})", if replace { "replaced" } else { "merged" }));
            Ok(report)
        }),
        ("POST", "/api/models/remove") => body(req).map_err(|e| (400, e)).and_then(|b| {
            studio.update_config(|next| remove_model(next, str_or(&b, "section", ""), str_or(&b, "name", "")).map_err(|e| (400, e)))?;
            Ok(studio.config_for_page())
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
            let (request, out) = crate::media::music::quantize_request(&cfg, &studio.root, &name, &quant).map_err(|e| (400, e))?;
            let job = studio.media.submit(crate::media::Kind::Music, request, name.clone(), format!("smaller copy ({quant})"), 1, 0.0, 200, None);
            let s = Arc::clone(studio);
            let id = job.id.clone();
            let reply = Json::obj([("job", Json::str(&job.id)), ("output", Json::str(out.to_string_lossy()))]);
            std::thread::spawn(move || {
                let Some(done) = s.media.wait(&id, std::time::Duration::from_secs(4 * 3600)) else { return };
                if done.status != "completed" {
                    return;
                }
                let set_it = s.update_config(|next| {
                    crate::models::registry::obj_mut(next, &["media", "music", "models", &name])
                        .map(|m| crate::util::set(m, "language_model", crate::models::detect::path_json(&out)))
                        .ok_or((404, "it is no longer in the settings".to_string()))
                });
                match set_it {
                    Ok(_) => s.log.push(format!("music model {name} now uses {}", out.display())),
                    Err((_, e)) => s.log.push(format!("music model {name}: could not use {}: {e}", out.display())),
                }
            });
            Ok(reply)
        }),
        // The viewer's Open in OAIY's window, which opens no other windows: the output in this computer's own viewer.
        // An output's file alone, and for a client on this computer alone (it shows on this screen).
        ("POST", "/api/open-file") => {
            if remote {
                Err((403, "an output opens on this computer for its own window alone".into()))
            } else {
                body(req).map_err(|e| (400, e)).and_then(|b| {
                    let rel = b.get("url").and_then(Json::as_str).and_then(output_path).ok_or((400, "not an output's file".to_string()))?;
                    let path = crate::media::resolve_file(&studio.output_root(), &rel).ok_or((404, "no such output file".to_string()))?;
                    crate::open_path(&path.to_string_lossy());
                    Ok(Json::obj([("opened", Json::str(path.to_string_lossy()))]))
                })
            }
        }
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

/// The part of an output's address after `/files/` (as the page has it: alone, or after this server's address), its
/// query or fragment left off.
fn output_path(url: &str) -> Option<String> {
    let path = url.split(['?', '#']).next()?;
    let path = path.strip_prefix("/files/").or_else(|| {
        let rest = path.strip_prefix("http://").or_else(|| path.strip_prefix("https://"))?;
        rest.split_once('/').and_then(|(_, p)| p.strip_prefix("files/"))
    })?;
    (!path.is_empty()).then(|| path.to_string())
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
            // (and the eGPU's stand-in, where it was this model: the configuration is refused while it names one
            // that is not listed, so the model could not be removed at all)
            if let Json::Obj(fields) = llm {
                if let Some((_, egpu)) = fields.iter_mut().find(|(k, _)| k == "egpu") {
                    if str_or(egpu, "fallback_model", "") == name {
                        crate::util::set(egpu, "fallback_model", Json::str(""));
                    }
                }
            }
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

/// The eGPU's server started by hand with `asked`, else the model it held (was loading, or failed to load), else the
/// first set to it. A server that has the model is left as it is, and one answering with another is not cut off:
/// `start` says so.
fn egpu_start(studio: &Arc<Studio>, cfg: &Json, asked: Option<String>) -> Result<(), String> {
    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    let again = studio.egpu.held().filter(|m| crate::llm::egpu::assigned(&llm, m));
    match asked.or(again).or_else(|| crate::llm::egpu::models(&llm).into_iter().next()) {
        Some(name) => studio.egpu.start(cfg, &studio.root, &name),
        None => Err("no model is set to run on the eGPU (Models), or it is switched off (Settings)".into()),
    }
}

/// Whether "the language model" is the eGPU's to hold: it is switched on and the default model is set to it
/// ([`crate::llm::holds_a_model`] says this computer's engine then holds none).
fn all_on_the_egpu(cfg: &Json) -> bool {
    cfg.get("llm").is_some_and(|llm| !crate::llm::holds_a_model(llm) && !crate::llm::egpu::models(llm).is_empty())
}

/// The page's "Language model" card's Start. This computer's engine's, unless the default model is set to run on the
/// eGPU: then the card's server's, with the default. Not that model here as well: the card showed "Stopped" over a model loaded on the eGPU, and its Start
/// loaded the 27B a second time, on a 24 GB Mac's own GPU, beside the card's copy. (This computer's engine left
/// failed, from standing in for a card that did not answer, is cleared by it: nothing else but Stop did.)
pub(crate) fn llm_start(studio: &Arc<Studio>, cfg: &Json) -> Result<(), String> {
    if !all_on_the_egpu(cfg) {
        return studio.llm.start(cfg, &studio.root);
    }
    let llm = cfg.get("llm").cloned().unwrap_or(Json::Null);
    if !crate::util::bool_or(&llm, "enabled", true) {
        return Err("the LLM is disabled in the configuration".into());
    }
    if studio.llm.state() == crate::llm::State::Failed {
        studio.llm.stop();
    }
    let default = Some(crate::llm::default_model(&llm)).filter(|m| crate::llm::egpu::assigned(&llm, m));
    egpu_start(studio, cfg, default)
}

/// The card's Stop: this computer's engine stopped, and where the default model is set to run on the eGPU, the card's
/// server too, which is where the model then is: a Stop that left it loaded there stopped nothing. Not while
/// the card is lent to a picture or a video in the making, whose worker is on the socket that server holds: the
/// model is not on the card then, and comes back when the job has it no more.
pub(crate) fn llm_stop(studio: &Arc<Studio>, cfg: &Json) {
    studio.llm.stop();
    let lent = cfg.get("llm").is_some_and(|llm| studio.egpu.status(llm).get("lent") == Some(&Json::Bool(true)));
    if all_on_the_egpu(cfg) && !lent {
        studio.egpu.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The viewer's Open names an output by its address, which the page has alone or after the server's.
    #[test]
    fn an_output_is_named_by_its_address() {
        assert_eq!(output_path("/files/images/a%20b.png").as_deref(), Some("images/a%20b.png"));
        assert_eq!(output_path("http://127.0.0.1:7860/files/images/a.png?v=2#x").as_deref(), Some("images/a.png"));
        assert_eq!(output_path("/v1/files/images/a.png"), None);
        assert_eq!(output_path("/files/"), None);
        assert_eq!(output_path("file:///etc/passwd"), None);
    }

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

    /// The Endpoints page's routes name every service a route can serve: one it lacked showed as Chat, its first
    /// choice, in that route's row (the speech, music, sound, 3D and picture tool routes did).
    #[test]
    fn the_route_table_offers_every_service() {
        let start = PAGE.find("const targets = [").expect("the page lists the services");
        let list = &PAGE[start..start + PAGE[start..].find("]];").expect("the list ends")];
        for target in crate::config::TARGETS {
            assert!(list.contains(&format!(r#"["{target}", ""#)), "{target} is not offered on the Endpoints page");
        }
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
        // The model that stands in for the eGPU's goes as that too: the configuration is refused while it names
        // a model that is not listed, so it could not be removed at all.
        let mut llm = cfg.get("llm").unwrap().clone();
        crate::util::set(&mut llm, "models", Json::parse(br#"[{"name":"big","path":"big.gguf","egpu":true},{"name":"small","path":"small.gguf"}]"#).unwrap());
        crate::util::set(&mut llm, "egpu", Json::parse(br#"{"enabled":true,"fallback_model":"small"}"#).unwrap());
        crate::util::set(&mut cfg, "llm", llm);
        config::validate(&cfg).unwrap();
        remove_model(&mut cfg, "llm", "small").unwrap();
        assert_eq!(cfg.get("llm").unwrap().get("egpu").unwrap().get("fallback_model").and_then(Json::as_str), Some(""));
        config::validate(&cfg).unwrap();
        remove_model(&mut cfg, "llm", "big").unwrap();
        // Sound effect and 3D models are removed the same way.
        for section in ["sound", "model3d"] {
            let media = crate::models::registry::obj_mut(&mut cfg, &["media", section]).unwrap();
            crate::util::set(media, "models", Json::parse(br#"{"m":{"path":"m"}}"#).unwrap());
            crate::util::set(media, "default_model", Json::str("m"));
            remove_model(&mut cfg, section, "m").unwrap();
            assert_eq!(cfg.get("media").unwrap().get(section).unwrap().get("default_model").and_then(Json::as_str), Some(""));
        }
        config::validate(&cfg).unwrap();
    }
}
