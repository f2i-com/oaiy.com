//! Pinned picture models (`media.pinned`): kept loaded in the picture worker on their GPU whatever the keep time, a
//! release or another job (one that would let a pinned model go is refused, saying which pin holds the GPU), and
//! loaded again as the studio starts and after the worker ends ([`keep`]). A pin that cannot load is shown failed, and
//! kept: it is tried again after [`RETRY`]. OAIY Network pins a worker's models this way (docs/NETWORK.md); the
//! language model's pin is `llm.pinned`.
//!
//! The picture worker is one process on one GPU, keeping MOSS-Image's and FLUX 3 Image Klein's models side by side
//! and Qwen Image's alone (`super::kept_beside`): what is pinned is what it can keep together, checked as a pin is
//! added ([`check`]). Pictures of other models are refused while it is pinned on their GPU, as are the media jobs that
//! need the GPU for a worker of their own (video, speech, music…): the pin is the person's choice of what the GPU is
//! for, and it is undone by unpinning, never by a job.
use super::{image_request, kept_beside, Kind, Media};
use crate::util::{random_id, str_or};
use crate::Studio;
use oaiy_engine::json::Json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The most models pinned at once.
pub const MAX_PINS: usize = 4;

/// A pin that failed to load is tried again after this.
const RETRY: Duration = Duration::from_secs(5 * 60);

/// How often the pins are looked after.
const EVERY: Duration = Duration::from_secs(3);

/// The picture a model is loaded with: small, its answer thrown away.
const WARM_PROMPT: &str = "A plain light grey square.";

/// Where a pinned model stands.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Being loaded, by the job named.
    Warming(String),
    /// Loaded in the picture worker.
    Ready,
    /// It could not be loaded, as said, at the time given; tried again after [`RETRY`].
    Failed(String, Instant),
}

/// The models pinned, in the order they were.
pub fn pinned(cfg: &Json) -> Vec<String> {
    cfg.get("media")
        .and_then(|m| m.get("pinned"))
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_str).map(str::to_owned).collect())
        .unwrap_or_default()
}

/// The family (`architecture`) of picture model `model`, where it is one of this studio's.
pub fn family(cfg: &Json, model: &str) -> Option<String> {
    let models = cfg.get("media")?.get("image")?.get("models")?;
    models.get(model).map(|m| str_or(m, "architecture", "qwen-image").to_string())
}

/// Whether `model` can be pinned beside the pins there are: one of this studio's picture models that the kept worker
/// serves (on WebGPU), the one model of its family, and one the worker keeps beside each other pin.
pub fn check(cfg: &Json, model: &str) -> Result<(), String> {
    let fam = family(cfg, model).ok_or_else(|| format!("{model} is not one of this computer's picture models"))?;
    if !matches!(fam.as_str(), "qwen-image" | "moss-image" | "flux3-klein") {
        return Err(format!("{model} cannot be pinned: its pictures are made by a worker started for each one"));
    }
    let backend = cfg.get("media").and_then(|m| m.get("image")).and_then(|i| i.get("models")).and_then(|m| m.get(model)).map(|m| str_or(m, "backend", "webgpu").to_string());
    if backend.as_deref().is_some_and(|b| b != "webgpu") {
        return Err(format!("{model} cannot be pinned: only WebGPU picture models are kept loaded"));
    }
    let others: Vec<String> = pinned(cfg).into_iter().filter(|m| m != model).collect();
    if others.len() >= MAX_PINS {
        return Err(format!("{MAX_PINS} models are pinned already: unpin one first"));
    }
    for other in &others {
        let Some(of) = family(cfg, other) else { continue };
        if of == fam {
            return Err(format!("{other} is pinned already, and the picture worker keeps one model of a family: unpin it first"));
        }
        if !kept_beside(&of, &fam) {
            return Err(format!("{model} is not kept beside {other}, which is pinned: unpin it first"));
        }
    }
    Ok(())
}

/// `cfg` with `model` pinned, or unpinned.
pub fn with(cfg: &Json, model: &str, pin: bool) -> Json {
    let mut next = cfg.clone();
    let mut list: Vec<String> = pinned(cfg).into_iter().filter(|m| m != model).collect();
    if pin {
        list.push(model.to_string());
    }
    let mut media = next.get("media").cloned().unwrap_or(Json::Obj(Vec::new()));
    crate::util::set(&mut media, "pinned", Json::Arr(list.into_iter().map(Json::Str).collect()));
    crate::util::set(&mut next, "media", media);
    next
}

/// The pins as `GET /api/resources` shows them: `{model, state, error?, job?}`, state `warming`, `ready`, `failed` or
/// `waiting` (not looked after yet, or waiting for the queue to empty).
pub fn snapshot(cfg: &Json, media: &Media) -> Json {
    let states = media.pin_states();
    Json::Arr(
        pinned(cfg)
            .into_iter()
            .map(|model| {
                let mut row = vec![("model", Json::str(&model))];
                match states.get(&model) {
                    Some(State::Ready) => row.push(("state", Json::str("ready"))),
                    Some(State::Warming(job)) => row.extend([("state", Json::str("warming")), ("job", Json::str(job))]),
                    Some(State::Failed(e, _)) => row.extend([("state", Json::str("failed")), ("error", Json::str(e))]),
                    None => row.push(("state", Json::str("waiting"))),
                }
                Json::obj(row)
            })
            .collect(),
    )
}

/// Look after the pins, for as long as the studio runs.
pub fn keep(studio: Arc<Studio>) {
    loop {
        std::thread::sleep(EVERY);
        tend(&studio);
    }
}

/// One look: each pin is ready where the kept worker holds it, its loading followed where it is loading, and the first
/// one neither loaded (nor failed lately) where no other job is waiting or running.
pub(crate) fn tend(studio: &Arc<Studio>) {
    let cfg = studio.config();
    let wanted = pinned(&cfg);
    let media = &studio.media;
    // (a kept worker that ended by itself is forgotten first: what it kept is loaded again)
    media.reap_ended();
    let kept: Vec<String> = media.kept_worker().map(|k| k.models).unwrap_or_default();
    // (the loading jobs looked up before the pins are locked, never with them: the runner locks the jobs and then the
    // kept worker, and the kept worker's users the pins)
    let loading: Vec<String> = media.pin_states().values().filter_map(|s| if let State::Warming(job) = s { Some(job.clone()) } else { None }).collect();
    let jobs: BTreeMap<String, Option<super::Job>> = loading.into_iter().map(|id| (id.clone(), media.get(&id))).collect();
    let mut next: Option<String> = None;
    {
        let mut states = media.pin_states();
        states.retain(|m, _| wanted.contains(m));
        for model in &wanted {
            let state = states.get(model).cloned();
            let now = match state.clone() {
                Some(State::Warming(job)) => match jobs.get(&job).cloned().flatten() {
                    Some(j) if !j.finished() => continue,
                    Some(j) if j.status == "completed" && kept.contains(model) => State::Ready,
                    Some(j) => State::Failed(j.error.unwrap_or_else(|| format!("its loading job {}", j.status)), Instant::now()),
                    None => State::Failed("its loading job was forgotten".into(), Instant::now()),
                },
                _ if kept.contains(model) => State::Ready,
                Some(State::Failed(e, at)) if at.elapsed() < RETRY => State::Failed(e, at),
                _ => {
                    if next.is_none() {
                        next = Some(model.clone());
                    }
                    continue;
                }
            };
            if matches!(now, State::Failed(..)) && !matches!(state, Some(State::Failed(..))) {
                if let State::Failed(e, _) = &now {
                    media.log.push(format!("pinned {model} could not be loaded: {e}"));
                    crate::resources::event("pin_failed", model, "", e);
                }
            }
            states.insert(model.clone(), now);
        }
    }
    // (loaded when nothing else is: a person's picture never waits for a pin's)
    let Some(model) = next else { return };
    if media.busy() || media.queued() > 0 {
        return;
    }
    // (still one that can be kept: its settings may have changed since it was pinned, its backend say)
    if let Err(e) = check(&cfg, &model) {
        media.log.push(format!("pinned {model} could not be loaded: {e}"));
        crate::resources::event("pin_failed", &model, "", &e);
        media.pin_states().insert(model, State::Failed(e, Instant::now()));
        return;
    }
    match warm(studio, &cfg, &model) {
        Ok(job) => {
            media.log.push(format!("loading pinned {model} ({job})"));
            crate::resources::event("pin_warm", &model, "", "pinned: loading it to keep it loaded");
            media.pin_states().insert(model, State::Warming(job));
        }
        Err(e) => {
            media.log.push(format!("pinned {model} could not be loaded: {e}"));
            crate::resources::event("pin_failed", &model, "", &e);
            media.pin_states().insert(model, State::Failed(e, Instant::now()));
        }
    }
}

/// A small picture of `model`, in a folder of its own that is forgotten with the job: the job that loads it.
fn warm(studio: &Arc<Studio>, cfg: &Json, model: &str) -> Result<String, String> {
    let scratch = studio.output_root().join(".incognito").join(random_id(""));
    let body = Json::obj([
        ("model", Json::str(model)),
        ("prompt", Json::str(WARM_PROMPT)),
        ("size", Json::str("512x512")),
        ("n", Json::Int(1)),
        ("seed", Json::Int(1)),
    ]);
    let (request, name, size, n) = image_request(cfg, &studio.root, &scratch, &body, false).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&scratch);
    })?;
    Ok(studio.media.submit(Kind::Image, request, name, size, n, 0.0, 200, Some(scratch)).id)
}

/// Pin states by model, as [`Media`] keeps them.
pub type States = BTreeMap<String, State>;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Json {
        let mut cfg = crate::config::default_json();
        let model = |arch: &str| {
            Json::obj(
                [("architecture", arch), ("transformer", "t.gguf"), ("text_encoder", "te.gguf"), ("vae", "vae.safetensors"), ("checkpoint", "xl.safetensors"), ("tokenizer", "tokenizer.json"), ("base", "qwen")]
                    .map(|(k, v)| (k, Json::str(v))),
            )
        };
        let image = crate::models::registry::obj_mut(&mut cfg, &["media", "image"]).unwrap();
        crate::util::set(
            image,
            "models",
            Json::obj([
                ("moss", model("moss-image")),
                ("moss-8", model("moss-image")),
                ("klein", model("flux3-klein")),
                ("qwen", model("qwen-image")),
                ("xl", model("sdxl")),
            ]),
        );
        cfg
    }

    /// What can be pinned is what the picture worker keeps together: a picture model of this studio's that it serves,
    /// one of its family, beside the others pinned; pinned twice it is listed once, and unpinned it is gone.
    #[test]
    fn a_pin_is_one_the_picture_worker_can_keep_beside_the_others() {
        let c = cfg();
        assert!(check(&c, "moss").is_ok());
        assert!(check(&c, "nope").unwrap_err().contains("not one of this computer's"));
        assert!(check(&c, "xl").unwrap_err().contains("started for each one"));
        let c = with(&c, "moss", true);
        assert_eq!(pinned(&c), vec!["moss"]);
        assert!(check(&c, "klein").is_ok(), "MOSS-Image's and Klein's are kept side by side");
        assert!(check(&c, "moss-8").unwrap_err().contains("one model of a family"));
        assert!(check(&c, "qwen").unwrap_err().contains("not kept beside moss"));
        assert!(check(&c, "moss").is_ok(), "pinning it again changes nothing");
        let c = with(&with(&c, "moss", true), "klein", true);
        assert_eq!(pinned(&c), vec!["moss", "klein"]);
        assert_eq!(crate::config::validate(&c), Ok(()));
        assert_eq!(pinned(&with(&c, "moss", false)), vec!["klein"]);
    }

    /// The settings take a short list of names, and nothing else.
    #[test]
    fn the_settings_take_a_short_list_of_names() {
        let mut c = cfg();
        for (bad, ok) in [
            (Json::str("moss"), false),
            (Json::Arr(vec![Json::Int(1)]), false),
            (Json::Arr(vec![Json::str(" ")]), false),
            (Json::Arr((0..=MAX_PINS).map(|i| Json::str(format!("m{i}"))).collect()), false),
            (Json::Arr(vec![Json::str("moss")]), true),
            (Json::Null, true),
        ] {
            let media = crate::models::registry::obj_mut(&mut c, &["media"]).unwrap();
            crate::util::set(media, "pinned", bad.clone());
            assert_eq!(crate::config::validate(&c).is_ok(), ok, "{}", bad.to_json());
        }
    }
}
